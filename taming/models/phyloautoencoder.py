import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.models.vqgan import VQModel

# from torchsummary import summary
from torchinfo import summary

PHYLOCONFIG_KEY = "phylomodel_params"
BASEMODEL_KEY = "basemodel"

class PhyloDisentangler(torch.nn.Module):
    def __init__(self, 
                in_channels, ch, out_ch, resolution, ## same ad ddconfigs for autoencoder
                embed_dim, n_embed, # same as codebook configs
                n_phylo_channels, n_phylolevels, codebooks_per_phylolevel, # The dimensions for the phylo descriptors.
                lossconfig): 
        super().__init__()

        self.ch = ch
        self.n_phylo_channels = n_phylo_channels

        self.loss = instantiate_from_config(lossconfig) # TODO: This is just a test. we will be putting the real phylo loss later

        # downsampling
        self.conv_in = nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            self.ch,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            nn.SiLU(),
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
            torch.nn.Conv2d(self.ch,
                            out_ch,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            nn.SiLU(),
        )

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

        return output, q_phylo_loss





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

        ckpt_path = args['ckpt_path']
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.freeze()

        self.phylo_disentangler = PhyloDisentangler(**phylo_args)

        # print model
        print('totalmodel', self)
        summary(self.cuda(), (1, 3, 512, 512))

    def encode(self, x):
        encoder_out = self.encoder(x)
        disentangler_out, phylo_quantizer_loss = self.phylo_disentangler(encoder_out)
        h = self.quant_conv(disentangler_out)
        quant, base_quantizer_loss, info = self.quantize(h)
        return quant, phylo_quantizer_loss, base_quantizer_loss, encoder_out, disentangler_out, info

    def forward(self, input):
        quant, phylo_quantizer_loss, base_quantizer_loss, encoder_out, disentangler_out, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, phylo_quantizer_loss, base_quantizer_loss, encoder_out, disentangler_out
    
    def forward2(self, input):
        encoder_out = self.encoder(input)
        h = self.quant_conv(encoder_out)
        quant, base_hypothetical_quantizer_loss, info = self.quantize(h)
        dec = self.decode(quant)
        return dec, base_hypothetical_quantizer_loss

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, phylo_quantizer_loss, base_quantizer_loss, encoder_out, disentangler_out = self(x)

        #TODO: will slow us down for no reason. maybe remove in the future.
        xrec_hypthetical, base_hypothetical_quantizer_loss = self.forward2(x)
        hypothetical_rec_loss =torch.mean(torch.abs(x.contiguous() - xrec_hypthetical.contiguous()))
        true_rec_loss = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        self.log("train/base_hypothetical_rec_loss", hypothetical_rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/base_hypothetical_quantizer_loss", base_hypothetical_quantizer_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
        self.log("train/base_true_rec_loss", true_rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/base_quantizer_loss", base_quantizer_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # autoencode
        aeloss, log_dict_ae = self.phylo_disentangler.loss(phylo_quantizer_loss, encoder_out, disentangler_out, 0, self.global_step, split="train")

        rec_loss = log_dict_ae["train/rec_loss"]
        self.log("train/disentangler_total_aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/disentangler_quantizer_loss", phylo_quantizer_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/disentangler_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, phylo_quantizer_loss, base_quantizer_loss, encoder_out, disentangler_out = self(x)
        aeloss, log_dict_ae = self.phylo_disentangler.loss(phylo_quantizer_loss, encoder_out, disentangler_out, 0, self.global_step, split="val")

        #TODO: will slow us down for no reason. maybe remove in the future.
        xrec_hypthetical, base_hypothetical_quantizer_loss = self.forward2(x)
        hypothetical_rec_loss =torch.mean(torch.abs(x.contiguous() - xrec_hypthetical.contiguous()))
        true_rec_loss = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        self.log("val/base_hypothetical_rec_loss", hypothetical_rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/base_hypothetical_quantizer_loss", base_hypothetical_quantizer_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/base_true_rec_loss", true_rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/base_quantizer_loss", base_quantizer_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)


        rec_loss = log_dict_ae["val/rec_loss"]
        phylo_quantizer_loss = log_dict_ae["val/quant_loss"]
        self.log("val/disentangler_rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/disentangler_total_aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/disentangler_quantizer_loss", phylo_quantizer_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
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