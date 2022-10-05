import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.models.vqgan import VQModel

# from torchsummary import summary
from torchinfo import summary
import collections

PHYLOCONFIG_KEY = "phylomodel_params"
BASEMODEL_KEY = "basemodel"

DISENTANGLER_DECODER_OUTPUT = 'output'
DISENTANGLER_ENCODER_INPUT = 'in'
DISENTANGLER_CLASS_OUTPUT = 'class'



class ClassifierLayer(torch.nn.Module):
    def __init__(self, num_of_inputs, num_of_outputs, num_of_layers = 1, bnorm=False, relu=True):
        super(ClassifierLayer, self).__init__()
        
        self.num_of_inputs = num_of_inputs

        l = [torch.nn.Flatten()] 
    
        for i in range(num_of_layers):
            if bnorm == True and i==0:
                l.append(torch.nn.BatchNorm1d(num_of_inputs))
            n_out = num_of_inputs if (i+1 != num_of_layers) else num_of_outputs
            l.append(torch.nn.Linear(num_of_inputs, n_out))
            if relu == True:
                l.append( torch.nn.ReLU())
            
        self.seq = torch.nn.Sequential(*l)

    def get_inputsize(self):
        return self.num_of_inputs

    
    def forward(self, input):
        return self.seq(input)



class PhyloDisentangler(torch.nn.Module):
    def __init__(self, 
                in_channels, ch, out_ch, resolution, ## same ad ddconfigs for autoencoder
                embed_dim, n_embed, # same as codebook configs
                n_phylo_channels, n_phylolevels, codebooks_per_phylolevel, # The dimensions for the phylo descriptors.
                lossconfig, lossconfig_phylo=None): 
        super().__init__()

        self.ch = ch
        self.n_phylo_channels = n_phylo_channels
        self.n_phylolevels = n_phylolevels
        self.codebooks_per_phylolevel = codebooks_per_phylolevel
        self.embed_dim = embed_dim

        self.loss = instantiate_from_config(lossconfig)

        # downsampling
        self.conv_in = nn.Sequential(
            nn.SiLU(),
            torch.nn.Conv2d(in_channels,
                            self.ch,
                            kernel_size=1,
                            stride=1,
                            padding=0),
        )

        # phylo MLP
        self.mlp_in = nn.Sequential(
            nn.LayerNorm([self.n_phylo_channels,resolution,resolution]),
            nn.Flatten(),
            nn.Linear(resolution*resolution*self.n_phylo_channels, n_phylolevels*codebooks_per_phylolevel*embed_dim),
            nn.SiLU(),
            nn.Unflatten(1, torch.Size([embed_dim, codebooks_per_phylolevel, n_phylolevels])),
        )
        # self.passthrough = nn.Identity()
        self.mlp_out = nn.Sequential(
            # nn.LayerNorm([in_channels,resolution,resolution]), # I removed this so we can use the codebooks exactly.
            nn.Flatten(),
            nn.Linear(n_phylolevels*codebooks_per_phylolevel*embed_dim, resolution*resolution*self.n_phylo_channels),
            nn.SiLU(),
            nn.Unflatten(1, torch.Size([self.n_phylo_channels,resolution,resolution])),
        )

        # quantizer
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        # upsampling
        self.conv_out = nn.Sequential(
            nn.SiLU(),
            torch.nn.Conv2d(self.ch,
                            out_ch,
                            kernel_size=1,
                            stride=1,
                            padding=0),
        )

        self.loss_phylo = None
        if lossconfig_phylo is not None:
            
            #TODO: maybe move logic here onward into its own function.

            # get loss and parse params
            self.loss_phylo = instantiate_from_config(lossconfig_phylo)
            output_size = self.loss_phylo.classifier_output_size
            num_fc_layers = self.loss_phylo.fc_layers
            len_features = n_phylolevels*codebooks_per_phylolevel*embed_dim
            assert n_phylolevels==len(self.loss_phylo.phylo_distances)+1, "Number of phylo distances should be consistent in the settings."



            # Create classification layers.
            self.classification_layers = {
                DISENTANGLER_CLASS_OUTPUT: ClassifierLayer(len_features, output_size, num_of_layers=num_fc_layers),
            }

            for indx, i in enumerate(self.loss_phylo.phylo_distances):
                level_name = str(i).replace(".", "")+"distance"

                self.classification_layers[level_name] = ClassifierLayer(
                        int((indx+1)*len_features/n_phylolevels), 
                        output_size, 
                        num_of_layers=num_fc_layers
                    )

            self.classification_layers = torch.nn.ModuleDict(self.classification_layers)


        # print model
        print('phylovqgan', self)
        summary(self.cuda(), (1, 256, 32, 32))
    
    def forward(self, input):
        h = self.conv_in(input)
        # print(h.shape, self.n_phylo_channels, self.ch)
        h_phylo, h_img = torch.split(h, [self.n_phylo_channels, self.ch - self.n_phylo_channels], dim=1)
        z_phylo = self.mlp_in(h_phylo)
        q_phylo, q_phylo_loss, info = self.quantize(z_phylo)
        z_q_phylo = self.mlp_out(q_phylo)
        h_ = torch.cat((z_q_phylo, h_img), 1)
        output = self.conv_out(h_)

        loss_dic = {'quantizer_loss': q_phylo_loss}
        outputs = {DISENTANGLER_DECODER_OUTPUT: output}

        # Phylo networks
        if self.loss_phylo is not None:
            for name, layer in self.classification_layers.items():
                num_of_levels_included = int(layer.get_inputsize()/(self.embed_dim * self.codebooks_per_phylolevel))
                outputs[name] = layer(q_phylo[:, :, :, :num_of_levels_included])
    
        return outputs, loss_dic   


class PhyloVQVAE(VQModel):
    def __init__(self, **args):
        print(args)
        
        # For wandb
        self.save_hyperparameters()
        # self.logger.experiment.config['lr'] =args['lr']
        # # add multiple parameters
        # wandb_logger.experiment.config.update({key1: val1, key2: val2})
        # # use directly wandb module
        # wandb.config["key"] = value
        # wandb.config.update()

        phylo_args = args[PHYLOCONFIG_KEY]
        del args[PHYLOCONFIG_KEY]

        super().__init__(**args)

        self.freeze()

        self.phylo_disentangler = PhyloDisentangler(**phylo_args)

        # print model
        # print('totalmodel', self)
        # summary(self.cuda(), (1, 3, 512, 512))

    def encode(self, x):
        encoder_out = self.encoder(x)
        #phylo_quantizer_loss, classification_phylo_loss
        disentangler_outputs, disentangler_loss_dic = self.phylo_disentangler(encoder_out)
        disentangler_out = disentangler_outputs[DISENTANGLER_DECODER_OUTPUT]
        h = self.quant_conv(disentangler_out)
        quant, base_quantizer_loss, info = self.quantize(h)

        #consolidate dicts
        base_loss_dic = {'quantizer_loss': base_quantizer_loss}
        in_out_disentangler = {
            DISENTANGLER_ENCODER_INPUT: encoder_out,
        }
        in_out_disentangler = {**in_out_disentangler, **disentangler_outputs}

        return quant, disentangler_loss_dic, base_loss_dic, in_out_disentangler, info

    def forward(self, input):
        quant, disentangler_loss_dic, base_loss_dic, in_out_disentangler, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, disentangler_loss_dic, base_loss_dic, in_out_disentangler
    
    #TODO: make this only for debugging.
    def forward_hypothetical(self, input):
        encoder_out = self.encoder(input)
        h = self.quant_conv(encoder_out)
        quant, base_hypothetical_quantizer_loss, info = self.quantize(h)
        dec = self.decode(quant)
        return dec, base_hypothetical_quantizer_loss

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, disentangler_loss_dic, base_loss_dic, in_out_disentangler = self(x)
        out_disentangler = {i:in_out_disentangler[i] for i in in_out_disentangler if i not in [DISENTANGLER_ENCODER_INPUT, DISENTANGLER_DECODER_OUTPUT]}

        #TODO: will slow us down for no reason. maybe remove in the future.
        xrec_hypthetical, base_hypothetical_quantizer_loss = self.forward_hypothetical(x)
        hypothetical_rec_loss =torch.mean(torch.abs(x.contiguous() - xrec_hypthetical.contiguous()))
        true_rec_loss = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        self.log("train/base_hypothetical_rec_loss", hypothetical_rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/base_hypothetical_quantizer_loss", base_hypothetical_quantizer_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
        self.log("train/base_true_rec_loss", true_rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/base_quantizer_loss", base_loss_dic['quantizer_loss'], prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # autoencode
        quantizer_disentangler_loss = disentangler_loss_dic['quantizer_loss']
        total_loss, log_dict_ae = self.phylo_disentangler.loss(quantizer_disentangler_loss, in_out_disentangler[DISENTANGLER_ENCODER_INPUT], in_out_disentangler[DISENTANGLER_DECODER_OUTPUT], 0, self.global_step, split="train")
        
        if self.phylo_disentangler.loss_phylo is not None:
            phylo_losses_dict = self.phylo_disentangler.loss_phylo(total_loss, out_disentangler, batch[DISENTANGLER_CLASS_OUTPUT])
            total_loss = phylo_losses_dict['cumulative_loss']

        rec_loss = log_dict_ae["train/rec_loss"]
        self.log("train/disentangler_total_aeloss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/disentangler_quantizer_loss", quantizer_disentangler_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/disentangler_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.phylo_disentangler.loss_phylo is not None:
            self.log("train/disentangler_phylo_loss", phylo_losses_dict['total_phylo_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, disentangler_loss_dic, base_loss_dic, in_out_disentangler = self(x)
        out_disentangler = {i:in_out_disentangler[i] for i in in_out_disentangler if i not in [DISENTANGLER_ENCODER_INPUT, DISENTANGLER_DECODER_OUTPUT]}

        quantizer_disentangler_loss = disentangler_loss_dic['quantizer_loss']
        total_loss, log_dict_ae = self.phylo_disentangler.loss(quantizer_disentangler_loss, in_out_disentangler[DISENTANGLER_ENCODER_INPUT], in_out_disentangler[DISENTANGLER_DECODER_OUTPUT], 0, self.global_step, split="val")
        
        if self.phylo_disentangler.loss_phylo is not None:
            phylo_losses_dict = self.phylo_disentangler.loss_phylo(total_loss, out_disentangler, batch[DISENTANGLER_CLASS_OUTPUT])
            total_loss = phylo_losses_dict['cumulative_loss']

        #TODO: will slow us down for no reason. maybe remove in the future.
        xrec_hypthetical, base_hypothetical_quantizer_loss = self.forward_hypothetical(x)
        hypothetical_rec_loss =torch.mean(torch.abs(x.contiguous() - xrec_hypthetical.contiguous()))
        true_rec_loss = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        self.log("val/base_hypothetical_rec_loss", hypothetical_rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/base_hypothetical_quantizer_loss", base_hypothetical_quantizer_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/base_true_rec_loss", true_rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/base_quantizer_loss", base_loss_dic['quantizer_loss'], prog_bar=True, logger=True, on_step=False, on_epoch=True)


        rec_loss = log_dict_ae["val/rec_loss"]
        # quant_loss = log_dict_ae["val/quant_loss"]
        self.log("val/disentangler_rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/disentangler_total_aeloss", total_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/disentangler_quantizer_loss", quantizer_disentangler_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        if self.phylo_disentangler.loss_phylo is not None:
            self.log("val/disentangler_phylo_loss", phylo_losses_dict['total_phylo_loss'], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        # self.log_dict(log_dict_ae)
        return self.log_dict

    
    def configure_optimizers(self):
        lr = self.learning_rate

        # for i in self.phylo_disentangler.parameters():
        #     print('helloooooooo', i.shape, i.requires_grad)
        # opt_ae = torch.optim.Adam(list(self.phylo_disentangler.encoder.parameters())+
        #                         list(self.phylo_disentangler.decoder.parameters())+
        #                         list(self.phylo_disentangler.quantize.parameters())+
        #                         list(self.phylo_disentangler.quant_conv.parameters())+
        #                         list(self.phylo_disentangler.post_quant_conv.parameters()),
        #                         lr=lr, betas=(0.5, 0.9))
        opt_ae = torch.optim.Adam(self.phylo_disentangler.parameters(), lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae], []