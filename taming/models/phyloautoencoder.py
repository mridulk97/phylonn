from taming.import_utils import instantiate_from_config
from taming.modules.losses.phyloloss import get_loss_name
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.models.vqgan import VQModel

from taming.analysis_utils import get_species_phylo_distance, get_HammingDistance_matrix, Embedding_Code_converter, aggregate_metric_from_specimen_to_species
from taming.plotting_utils import plot_heatmap, dump_to_json, plot_confusionmatrix

# from torchsummary import summary
from torchinfo import summary

import taming.constants as CONSTANTS
import itertools

from torchmetrics import F1Score

import math

def get_hidden_layer_sizes(num_of_inputs, num_of_outputs, num_of_layers): 
    out_sizes = []
    if num_of_layers > 1:
        diff = num_of_inputs-num_of_outputs
        for i in range(num_of_layers-1):
            num_of_hidden = int(num_of_inputs - (i+1)*diff/num_of_layers)
            out_sizes.append(num_of_hidden)
    out_sizes.append(num_of_outputs)
    return out_sizes


class ClassifierLayer(torch.nn.Module):
    def __init__(self, num_of_inputs, num_of_outputs, num_of_layers = 1): # bnorm=False,
        super(ClassifierLayer, self).__init__()
        
        self.num_of_inputs = num_of_inputs
        
        out_sizes = get_hidden_layer_sizes(num_of_inputs, num_of_outputs, num_of_layers)

        l = [torch.nn.Flatten()] 
    
        for i in range(num_of_layers):            
            n_out = out_sizes[i]
            n_in = out_sizes[i-1] if i>0 else num_of_inputs
            
            l.append(torch.nn.Linear(n_in, n_out))
            if (i!=num_of_layers-1):
                l.append(torch.nn.ReLU())
            
        self.seq = torch.nn.Sequential(*l)

    def get_inputsize(self):
        return self.num_of_inputs

    
    def forward(self, input):
        return self.seq(input)
    
    
def make_MLP(input_dim, output_dim, num_of_layers = 1, normalize=False):        
        flattened_input_dim = math.prod(input_dim)
        flattened_output_dim = math.prod(output_dim)
        
        out_sizes = get_hidden_layer_sizes(flattened_input_dim, flattened_output_dim, num_of_layers)
        
        l =[]
        if normalize:
            l.append(nn.LayerNorm(input_dim))
        l.append(nn.Flatten())
        
        in_ = flattened_input_dim 
        for i in range(num_of_layers):
            l = l + [nn.Linear(in_, out_sizes[i]),
                nn.SiLU(),
            ]
            in_ = out_sizes[i]
            
        # Remove RELU from last layer
        l = l[:-1]
            
        l.append(nn.Unflatten(1, torch.Size(output_dim)))
        
        return torch.nn.Sequential(*l)
        

# output_sizes are ordered such that we start with highest ancestor and move down.
def create_phylo_classifier_layers(len_features, output_sizes, num_fc_layers, n_phylolevels, phylo_distances):
    classification_layers = {
        CONSTANTS.DISENTANGLER_CLASS_OUTPUT: ClassifierLayer(len_features, output_sizes[-1], num_of_layers=num_fc_layers),
    }

    for indx, i in enumerate(phylo_distances):
        level_name = get_loss_name(phylo_distances, indx)
        
        out_size = output_sizes[indx]

        classification_layers[level_name] = ClassifierLayer(
                int((indx+1)*len_features/n_phylolevels), 
                out_size, 
                num_of_layers=num_fc_layers
            )

    return torch.nn.ModuleDict(classification_layers)

class PhyloDisentangler(torch.nn.Module):
    def __init__(self, 
                in_channels, ch, out_ch, resolution, ## same ad ddconfigs for autoencoder
                embed_dim, n_embed, # same as codebook configs
                n_phylo_channels, n_phylolevels, codebooks_per_phylolevel, # The dimensions for the phylo descriptors.
                lossconfig, 
                n_mlp_layers=1, n_levels_non_attribute=None,
                lossconfig_phylo=None, lossconfig_kernelorthogonality=None, lossconfig_anticlassification=None, verbose=False): 
        super().__init__()

        self.ch = ch
        self.n_phylo_channels = n_phylo_channels
        self.n_phylolevels = n_phylolevels
        self.codebooks_per_phylolevel = codebooks_per_phylolevel
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.n_levels_non_attribute = n_levels_non_attribute
        self.n_mlp_layers = n_mlp_layers

        self.verbose = verbose

        self.loss = instantiate_from_config(lossconfig)

        # downsampling
        out_sizes = get_hidden_layer_sizes(in_channels, self.ch, n_mlp_layers)
        out_sizes.append(self.ch)
        l = []
        _in = in_channels
        for i in range(n_mlp_layers):
            l.append(torch.nn.Conv2d(_in,
                out_sizes[i],
                kernel_size=1,
                stride=1,
                padding=0))
            l.append(nn.SiLU())
            _in = out_sizes[i]
        self.conv_in = nn.Sequential(*l)
        
        # phylo MLP
        self.mlp_in = make_MLP([self.n_phylo_channels,resolution,resolution], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=True)
        self.mlp_out = make_MLP([embed_dim, codebooks_per_phylolevel, n_phylolevels], [self.n_phylo_channels,resolution,resolution], n_mlp_layers, normalize=False)
        
        self.mlp_in_non_attribute = make_MLP([self.ch - self.n_phylo_channels,resolution,resolution], [embed_dim, codebooks_per_phylolevel, n_levels_non_attribute], n_mlp_layers, normalize=True)
        self.mlp_out_non_attribute = make_MLP([embed_dim, codebooks_per_phylolevel, n_levels_non_attribute], [self.ch - self.n_phylo_channels,resolution,resolution], n_mlp_layers, normalize=False)
            

        # quantizer
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.embedding_converter = Embedding_Code_converter(self.quantize.get_codebook_entry_index, self.quantize.embedding, (self.embed_dim, self.codebooks_per_phylolevel, self.n_phylolevels))


        # upsampling
        out_sizes = get_hidden_layer_sizes(self.ch, out_ch, n_mlp_layers)
        out_sizes.append(out_ch)
        l = []
        _in = self.ch
        for i in range(n_mlp_layers):
            l.append(nn.SiLU())
            l.append(torch.nn.Conv2d(_in,
                out_sizes[i],
                kernel_size=1,
                stride=1,
                padding=0))
            _in = out_sizes[i]
        self.conv_out = nn.Sequential(*l)

        self.loss_kernelorthogonality = None
        if lossconfig_kernelorthogonality is not None:
            lossconfig_kernelorthogonality['params'] = {**lossconfig_kernelorthogonality['params'], **{'verbose': verbose}}
            self.loss_kernelorthogonality= instantiate_from_config(lossconfig_kernelorthogonality)



        self.loss_phylo = None
        if lossconfig_phylo is not None:
            # get loss and parse params
            lossconfig_phylo['params'] = {**lossconfig_phylo['params'], **{'verbose': verbose}}
            self.loss_phylo = instantiate_from_config(lossconfig_phylo)
            num_fc_layers = self.loss_phylo.fc_layers
            len_features = n_phylolevels*codebooks_per_phylolevel*embed_dim
            assert n_phylolevels==len(self.loss_phylo.phylo_distances)+1, "Number of phylo distances should be consistent in the settings."
            
            # Create classification layers.
            self.classification_layers = create_phylo_classifier_layers(len_features, self.loss_phylo.classifier_output_sizes, num_fc_layers, n_phylolevels, self.loss_phylo.phylo_distances)
            
   
        # Create anti-classification
        self.loss_anticlassification = None
        if lossconfig_anticlassification is not None:
            lossconfig_anticlassification['params'] = {**lossconfig_anticlassification['params'], **{'verbose': verbose}}
            self.loss_anticlassification= instantiate_from_config(lossconfig_anticlassification)
            self.codebook_mapping_layers = make_MLP([embed_dim, codebooks_per_phylolevel, n_levels_non_attribute], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=False)


        # print model
        print('phylovqgan', self)
        summary(self.cuda(), (1, in_channels, resolution, resolution))
    
    
    # NOTE: This does not return losses. Only used for outputting!
    def from_quant_only(self, quant_attribute, quant_nonattribute=None):        
        hout_phylo = self.mlp_out(quant_attribute)
        
        hout_non_phylo = self.mlp_out_non_attribute(quant_nonattribute)
        h_ = torch.cat((hout_phylo, hout_non_phylo), 1)
            
        output = self.conv_out(h_)
        
        outputs = {CONSTANTS.DISENTANGLER_DECODER_OUTPUT: output}
        
        return outputs, {}
    
    def encode(self, input, overriding_quant_attr=None, overriding_quant_nonattr=None):
        h = self.conv_in(input)

        # print(h.shape, self.n_phylo_channels, self.ch)
        h_phylo, h_img = torch.split(h, [self.n_phylo_channels, self.ch - self.n_phylo_channels], dim=1)
        z_phylo = self.mlp_in(h_phylo)
        zq_phylo, q_phylo_loss, info_attr = self.quantize(z_phylo)

        if overriding_quant_attr is not None:
            assert zq_phylo.shape == overriding_quant_attr.shape, str(zq_phylo.shape) + "!=" + str(overriding_quant_attr.shape)
            zq_phylo = overriding_quant_attr
        
        
        loss_dic = {'quantizer_loss': q_phylo_loss}
        outputs = {CONSTANTS.QUANTIZED_PHYLO_OUTPUT: zq_phylo}

        z_nonphylo = self.mlp_in_non_attribute(h_img)
        zq_nonphylo, q_nonphylo_loss, info_nonattr = self.quantize(z_nonphylo)
        if overriding_quant_nonattr is not None:
            assert z_nonphylo.shape == overriding_quant_nonattr.shape, str(z_nonphylo.shape) + "!=" + str(overriding_quant_nonattr.shape)
            z_nonphylo = overriding_quant_attr
        
        loss_dic = {'quantizer_loss': q_phylo_loss + q_nonphylo_loss}
                
        if self.loss_kernelorthogonality is not None:
            kernel_orthogonality_loss = 0
            for i in range(self.n_mlp_layers):
                kernel_orthogonality_loss = kernel_orthogonality_loss + self.loss_kernelorthogonality(self.conv_in[i*2].weight)
            loss_dic['kernel_orthogonality_loss'] = kernel_orthogonality_loss    
                            
        return zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, info_attr, info_nonattr
    
    def decode(self, zq_phylo, zq_nonphylo, loss_dic={}, outputs={}, h_img=None):
        hout_phylo = self.mlp_out(zq_phylo)
            
        hout_non_phylo = self.mlp_out_non_attribute(zq_nonphylo)
        h_ = torch.cat((hout_phylo, hout_non_phylo), 1)
        
        if self.loss_anticlassification is not None:
            mapping_loss, learning_loss = self.loss_anticlassification(self.codebook_mapping_layers, zq_nonphylo, zq_phylo)
        
            loss_dic['anti_classification_mapping_loss'] = mapping_loss
            loss_dic['anti_classification_learning_loss'] = learning_loss            

        output = self.conv_out(h_)
        
        outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]= output
        if self.loss_anticlassification is not None:
            outputs[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] = zq_nonphylo
            

        # Phylo networks
        if self.loss_phylo is not None:
            outputs[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT] = self.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](self.codebook_mapping_layers(zq_nonphylo))

            for name, layer in self.classification_layers.items():
                num_of_levels_included = int(layer.get_inputsize()/(self.embed_dim * self.codebooks_per_phylolevel))
                outputs[name] = layer(zq_phylo[:, :, :, :num_of_levels_included]) # 0 for level 1, 0:1 for level 2, etc.
                
        return outputs, loss_dic


    
    def forward(self, input, overriding_quant_attr=None, overriding_quant_nonattr=None):
        zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, _, _ = self.encode(input, overriding_quant_attr, overriding_quant_nonattr)
        outputs, loss_dic = self.decode(zq_phylo, zq_nonphylo, loss_dic, outputs, h_img)
    
        return outputs, loss_dic   

#*********************************



class PhyloVQVAE(VQModel):
    def __init__(self, **args):
        print(args)
        
        # self.automatic_optimization = False
        
        # For wandb
        self.save_hyperparameters()

        phylo_args = args[CONSTANTS.PHYLOCONFIG_KEY]
        del args[CONSTANTS.PHYLOCONFIG_KEY]
        if CONSTANTS.DISENTANGLERTYPE_KEY in args:
            del args[CONSTANTS.DISENTANGLERTYPE_KEY]

        super().__init__(**args)

        self.freeze()
 
        self.phylo_disentangler = PhyloDisentangler(**phylo_args)

        self.verbose = phylo_args.get('verbose', False)
        
    
    def encode(self, x, overriding_quant=None, overriding_quant_nonattr=None):
        encoder_out = self.encoder(x)
        zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, info_attr, info_nonattr = self.phylo_disentangler.encode(encoder_out, overriding_quant, overriding_quant_nonattr)
        return zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, encoder_out, info_attr, info_nonattr
    
    def decode(self, zq_phylo, zq_nonphylo, loss_dic={}, outputs={}, h_img=None, encoder_out=None):
        disentangler_outputs, disentangler_loss_dic = self.phylo_disentangler.decode(zq_phylo, zq_nonphylo, loss_dic, outputs, h_img)
        
        disentangler_out = disentangler_outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]
        h = self.quant_conv(disentangler_out)
        quant, base_quantizer_loss, _ = self.quantize(h)

        #consolidate dicts
        base_loss_dic = {'quantizer_loss': base_quantizer_loss}
        in_out_disentangler = {
            CONSTANTS.DISENTANGLER_ENCODER_INPUT: encoder_out,
        }
        in_out_disentangler = {**in_out_disentangler, **disentangler_outputs}
        
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec, disentangler_loss_dic, base_loss_dic, in_out_disentangler
    
    def forward(self, input, overriding_quant=None, overriding_quant_nonattr=None):
        zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, encoder_out, _, _ = self.encode(input, overriding_quant, overriding_quant_nonattr)
        dec, disentangler_loss_dic, base_loss_dic, in_out_disentangler = self.decode(zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, encoder_out)
        return dec, disentangler_loss_dic, base_loss_dic, in_out_disentangler    
    
    #NOTE: This does not return losses. Only used for outputting!
    def from_quant_only(self, quant, quant_nonattribute=None):
        disentangler_outputs, _ = self.phylo_disentangler.from_quant_only(quant, quant_nonattribute)
        disentangler_out = disentangler_outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]
        h = self.quant_conv(disentangler_out)
        quant, _, _ = self.quantize(h)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec, {}
        
    
    def forward_hypothetical(self, input):
        encoder_out = self.encoder(input)
        h = self.quant_conv(encoder_out)
        quant, base_hypothetical_quantizer_loss, info = self.quantize(h)
        dec, _, _, _ = self.decode(quant)
        return dec, base_hypothetical_quantizer_loss

    def step(self, batch, batch_idx, optimizer_idx, prefix):        
        x = self.get_input(batch, self.image_key)
        xrec, disentangler_loss_dic, base_loss_dic, in_out_disentangler = self(x)
        out_class_disentangler = {i:in_out_disentangler[i] for i in in_out_disentangler if i not in CONSTANTS.NON_CLASS_TENSORS}

        if optimizer_idx==0 or (self.phylo_disentangler.loss_anticlassification is None):
            losses = {}
            # base losses
            true_rec_loss = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
            # self.log(prefix+"/base_true_rec_loss", true_rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            # self.log(prefix+"/base_quantizer_loss", base_loss_dic['quantizer_loss'], prog_bar=False, logger=True, on_step=False, on_epoch=True)
            losses[prefix+"/base_true_rec_loss"] = true_rec_loss
            losses[prefix+"/base_quantizer_loss"] = base_loss_dic['quantizer_loss']
            
            # autoencode
            quantizer_disentangler_loss = disentangler_loss_dic['quantizer_loss']
            total_loss, log_dict_ae = self.phylo_disentangler.loss(quantizer_disentangler_loss, in_out_disentangler[CONSTANTS.DISENTANGLER_ENCODER_INPUT], in_out_disentangler[CONSTANTS.DISENTANGLER_DECODER_OUTPUT], 0, self.global_step, split=prefix)


                        
            if self.phylo_disentangler.loss_kernelorthogonality is not None:
                kernelorthogonality_disentangler_loss = disentangler_loss_dic['kernel_orthogonality_loss']
                total_loss = total_loss + kernelorthogonality_disentangler_loss*self.phylo_disentangler.loss_kernelorthogonality.weight
                # self.log(prefix+"/disentangler_kernelorthogonality_loss", kernelorthogonality_disentangler_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
                losses[prefix+"/disentangler_kernelorthogonality_loss"] = kernelorthogonality_disentangler_loss

            
            
            

            if self.phylo_disentangler.loss_phylo is not None:
                phylo_losses_dict = self.phylo_disentangler.loss_phylo(total_loss, out_class_disentangler, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                total_loss = phylo_losses_dict['cumulative_loss']
                # self.log(prefix+CONSTANTS.DISENTANGLER_PHYLO_LOSS, phylo_losses_dict['total_phylo_loss'], prog_bar=self.verbose, logger=True, on_step=False, on_epoch=True)
                losses[prefix+CONSTANTS.DISENTANGLER_PHYLO_LOSS] = phylo_losses_dict['total_phylo_loss']
                
                for i in phylo_losses_dict['individual_losses']:
                    # self.log(prefix+"/disentangler_phylo_"+i, phylo_losses_dict['individual_losses'][i], prog_bar=True, logger=True, on_step=False, on_epoch=True)
                    losses[prefix+"/disentangler_phylo_"+i] = phylo_losses_dict['individual_losses'][i]

                for i in phylo_losses_dict:
                    if "_f1" in i:
                        # self.log(prefix+"/disentangler_phylo_"+i, phylo_losses_dict[i], prog_bar=True, logger=True, on_step=False, on_epoch=True)
                        losses[prefix+"/disentangler_phylo_"+i] = phylo_losses_dict[i]
            
            
            
            

            
            
            if self.phylo_disentangler.loss_anticlassification is not None:
                learning_loss = disentangler_loss_dic['anti_classification_learning_loss']
                total_loss = total_loss + learning_loss*self.phylo_disentangler.loss_anticlassification.weight
                
                if self.phylo_disentangler.loss_phylo:
                    anti_classification_classifier_output = in_out_disentangler[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT]
                    anti_classification_f1_score = self.phylo_disentangler.loss_phylo.F1(anti_classification_classifier_output, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                    # self.log(prefix+"/anti_classification_classifier_output", anti_classification_f1_score, prog_bar=False, logger=True, on_step=False, on_epoch=True)
                    losses[prefix+"/anti_classification_classifier_output"] = anti_classification_f1_score





            # if self.verbose:
            with torch.no_grad():
                _, _, _, in_out_disentangler_of_rec = self(xrec)
                rec_classification = in_out_disentangler_of_rec[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
                generated_f1_score = self.phylo_disentangler.loss_phylo.F1(rec_classification, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                # self.log(prefix+"/generated_f1_score", generated_f1_score, prog_bar=False, logger=True, on_step=False, on_epoch=True)
                losses[prefix+"/generated_f1_score"] = generated_f1_score
            




            # self.log(prefix+"/disentangler_total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            losses[prefix+"/disentangler_total_loss"] = total_loss




            rec_loss = log_dict_ae[prefix+"/rec_loss"]
            # self.log(prefix+"/disentangler_quantizer_loss", quantizer_disentangler_loss, prog_bar=True, logger=True, on_step=self.verbose, on_epoch=True)
            # self.log(prefix+"/disentangler_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)        
            losses[prefix+"/disentangler_quantizer_loss"] = quantizer_disentangler_loss
            losses[prefix+"/disentangler_rec_loss"] = rec_loss
            
            
            
            self.log_dict(losses, logger=True, on_step=False, on_epoch=True)
            
            # if prefix == 'train':
            #     return total_loss
            # else:
            outputs = {
                'loss': total_loss,
                
                CONSTANTS.QUANTIZED_PHYLO_OUTPUT: in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT],
                CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT: in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT],
                CONSTANTS.DISENTANGLER_CLASS_OUTPUT: batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT],
                CONSTANTS.DATASET_CLASSNAME: batch[CONSTANTS.DATASET_CLASSNAME],
                'logs': losses
            }
            
                
                # returned = {**{'log':losses}, **outputs}
            return outputs
                # return total_loss
                # return {
                #     'loss': total_loss,
                #     CONSTANTS.QUANTIZED_PHYLO_OUTPUT: in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT],
                #     CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT: in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT],
                #     CONSTANTS.DISENTANGLER_CLASS_OUTPUT: batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT],
                #     CONSTANTS.DATASET_CLASSNAME: batch[CONSTANTS.DATASET_CLASSNAME],
                # }
        
        if optimizer_idx==1 and (self.phylo_disentangler.loss_anticlassification is not None):
            mapping_loss = disentangler_loss_dic['anti_classification_mapping_loss']
            total_loss = mapping_loss*self.phylo_disentangler.loss_anticlassification.beta*self.phylo_disentangler.loss_anticlassification.weight
            # self.log(prefix+"/disentangler_anti_classification_loss", mapping_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

            # if prefix == 'train':
            #     return total_loss
            # else:
            # self.log_dict({
            #         prefix+"/disentangler_anti_classification_loss": mapping_loss
            #     }, logger=True, on_step=True, on_epoch=True)
            # return total_loss
            # if prefix == 'train':
            #     return total_loss
            # else:
            losses = {prefix+"/disentangler_anti_classification_loss": mapping_loss}
            
            return {
                'loss': total_loss,
                'logs': losses
                # 'log': {
                #     prefix+"/disentangler_anti_classification_loss": mapping_loss
                # }
            }
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        outputs = self.step(batch, batch_idx, optimizer_idx=optimizer_idx, prefix='train')
        self.log_dict(outputs['logs'], logger=True, on_step=False, on_epoch=True)
        return outputs['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        outputs = self.step(batch, batch_idx, optimizer_idx=0, prefix='val')
        self.log_dict(outputs['logs'], logger=True, on_step=False, on_epoch=True)
        del outputs['logs']
        return outputs
        
    #NOTE: Thois can be added with no problems to logging    
    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        if CONSTANTS.QUANTIZED_PHYLO_OUTPUT in outputs[0]:
            self.validation_epoch_end_zq_phylos = torch.cat([x[CONSTANTS.QUANTIZED_PHYLO_OUTPUT] for x in outputs], 0)
            self.validation_epoch_end_zq_nonphylos = torch.cat([x[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] for x in outputs], 0)
            self.validation_epoch_end_classes = torch.cat([x[CONSTANTS.DISENTANGLER_CLASS_OUTPUT] for x in outputs], 0)
            self.validation_epoch_end_classnames = list(itertools.chain.from_iterable([x[CONSTANTS.DATASET_CLASSNAME] for x in outputs]))
        

    ##################### test ###########
        
    # NOTE: This is kinda hacky. But ok for now for test purposes.
    def set_test_chkpt_path(self, chkpt_path):
        self.test_chkpt_path = chkpt_path

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        rec_x, _, _, in_out_disentangler = self(x)

        pred_ = in_out_disentangler[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
        class_ = batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
        classname_ = batch[CONSTANTS.DATASET_CLASSNAME]
        zq_phylo_features = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
        
        _, _, _, in_out_disentangler_rec = self(rec_x)
        class_recreated = in_out_disentangler_rec[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
        
        return {
            'pred': pred_, 
            'pred_rec': class_recreated,
            CONSTANTS.DISENTANGLER_CLASS_OUTPUT: class_, #.unsqueeze(-1)
            CONSTANTS.DATASET_CLASSNAME: classname_,
            'zq_phylo': zq_phylo_features
        }
    
    @torch.no_grad()
    def test_epoch_end(self, in_out):
        assert self.phylo_disentangler.loss_phylo is not None, "testing only enabled when there is phyloloss"
        test_measures = {}

        preds =torch.cat([x['pred'] for x in in_out], 0)
        pred_rec =torch.cat([x['pred_rec'] for x in in_out], 0)
        classes = torch.cat([x[CONSTANTS.DISENTANGLER_CLASS_OUTPUT] for x in in_out], 0)
        classnames = list(itertools.chain.from_iterable([x[CONSTANTS.DATASET_CLASSNAME] for x in in_out]))
        zq_phylos = torch.cat([x['zq_phylo'] for x in in_out], 0)
        sorting_indices = numpy.argsort(classes.cpu())
        sorted_zq_phylos = zq_phylos[sorting_indices, :]
        sorted_zq_phylos_codes = self.phylo_disentangler.embedding_converter.get_phylo_codes(sorted_zq_phylos)
        reverse_shaped_sorted_zq_phylos_codes = self.phylo_disentangler.embedding_converter.reshape_code(sorted_zq_phylos_codes, reverse=True)
        sorted_class_names_according_to_class_indx = [classnames[i] for i in sorting_indices]
        unique_sorted_class_names_according_to_class_indx = sorted(set(sorted_class_names_according_to_class_indx))

        F1 = F1Score(num_classes=self.phylo_disentangler.loss_phylo.classifier_output_sizes[-1], multiclass=True).to(preds.device)

        print("calculate F1 and Confusion matrix")
        test_measures['class_f1'] = F1(preds, classes).item()
        test_measures['recreated_class_f1'] = F1(pred_rec, classes).item()
        dump_to_json(test_measures, self.test_chkpt_path)

        plot_confusionmatrix(preds, classes, unique_sorted_class_names_according_to_class_indx, self.test_chkpt_path, CONSTANTS.TEST_DIR, "Confusion Matrix of real images")
        plot_confusionmatrix(pred_rec, classes, unique_sorted_class_names_according_to_class_indx, self.test_chkpt_path, CONSTANTS.TEST_DIR, "Confusion Matrix of recreated images")
        
        num_of_levels = self.phylo_disentangler.n_phylolevels
        for level in range(num_of_levels):
        
            #****************

            # plots per specimen
            print("Calculating embedding distances for level {}".format(level))
            
            sub_sorted_zq_phylos_codes = self.phylo_disentangler.embedding_converter.reshape_code(reverse_shaped_sorted_zq_phylos_codes[:, :, :level+1])
            zq_hamming_distances = get_HammingDistance_matrix(sub_sorted_zq_phylos_codes)
            
            plot_heatmap(zq_hamming_distances.cpu(), self.test_chkpt_path, title='zq hamming distances for level {}'.format(level), postfix=CONSTANTS.TEST_DIR)
            
            #********************
            print("Calculating phylo distances for level {}".format(level))
            
            level_relative_distance = self.phylo_disentangler.loss_phylo.get_relative_distance_for_level(level)
            species_distances = get_species_phylo_distance(sorted_class_names_according_to_class_indx, self.phylo_disentangler.loss_phylo.phylogeny.get_distance_between_parents, relative_distance=level_relative_distance)
            plot_heatmap(species_distances, self.test_chkpt_path, title='phylo distances for level {}'.format(level), postfix=CONSTANTS.TEST_DIR)
            
            #******************
            print("Calculating aggregated distances for level {}".format(level))
            # plot per species
            
            #TODO: This is not optimal since we are going from species -> specimen -> species again. Could be optimized.
            phylo_dist = aggregate_metric_from_specimen_to_species(sorted_class_names_according_to_class_indx, species_distances)
            plot_heatmap(phylo_dist.cpu(), self.test_chkpt_path, title='phylo species distances for level {}'.format(level), postfix=CONSTANTS.TEST_DIR)
            
            embedding_dist = aggregate_metric_from_specimen_to_species(sorted_class_names_according_to_class_indx, zq_hamming_distances)
            plot_heatmap(embedding_dist.cpu(), self.test_chkpt_path, title='zq hamming species distances for level {}'.format(level), postfix=CONSTANTS.TEST_DIR)
            

        return test_measures
    #######################################

    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(self.phylo_disentangler.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_mapping = torch.optim.Adam(self.phylo_disentangler.codebook_mapping_layers.parameters(), lr=lr, betas=(0.5, 0.9))
        
        lr_schedulers = [{
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_ae, 150, eta_min=lr*0.01),
            "monitor": "val"+CONSTANTS.DISENTANGLER_PHYLO_LOSS
            },{
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_ae, 150, eta_min=lr*0.01),
            "monitor": "val"+CONSTANTS.DISENTANGLER_PHYLO_LOSS
            },
        ]
        
        return [opt_ae, opt_mapping], lr_schedulers 