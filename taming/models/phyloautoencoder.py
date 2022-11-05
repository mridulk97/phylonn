from ast import Constant
from asyncio import constants
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy

from main import instantiate_from_config

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






class ClassifierLayer(torch.nn.Module):
    def __init__(self, num_of_inputs, num_of_outputs, num_of_layers = 1, relu=True, relu_last_layer=True, repeatInput=True): # bnorm=False,
        super(ClassifierLayer, self).__init__()
        
        self.num_of_inputs = num_of_inputs

        l = [torch.nn.Flatten()] 
    
        for i in range(num_of_layers):
            # if bnorm == True and i==0:
            #     l.append(torch.nn.BatchNorm1d(num_of_inputs))
            n_out = (num_of_inputs if (i+1 != num_of_layers) else num_of_outputs) if repeatInput else num_of_outputs
            n_in = num_of_inputs if repeatInput else (num_of_inputs if (i == 0) else num_of_outputs)
            l.append(torch.nn.Linear(n_in, n_out))
            is_last_layer = (i==num_of_layers-1)
            is_last_layer_but_dont_use_relu = is_last_layer and not relu_last_layer
            print(i, num_of_layers, is_last_layer_but_dont_use_relu)
            if relu and not is_last_layer_but_dont_use_relu:
                l.append( torch.nn.ReLU())
            
        self.seq = torch.nn.Sequential(*l)

    def get_inputsize(self):
        return self.num_of_inputs

    
    def forward(self, input):
        return self.seq(input)
    
    
def make_MLP(input_dim, output_dim, num_of_layers = 1, normalize=False):        
        flattened_input_dim = math.prod(input_dim)
        flattened_output_dim = math.prod(output_dim)
        
        l =[]
        if normalize:
            l.append(nn.LayerNorm(input_dim))
        l.append(nn.Flatten())
        
        in_ = flattened_input_dim 
        for i in range(num_of_layers):
            l = l + [nn.Linear(in_, flattened_output_dim),
                nn.SiLU(),
            ]
            in_ = flattened_output_dim
            
        l.append(nn.Unflatten(1, torch.Size(output_dim)))
        
        return torch.nn.Sequential(*l)
        

# output_sizes are ordered such that we start with highest ancestor and move down.
def create_phylo_classifier_layers(len_features, output_sizes, num_fc_layers, n_phylolevels, phylo_distances, repeatInput=True, relu_last_layer=True):
    classification_layers = {
        CONSTANTS.DISENTANGLER_CLASS_OUTPUT: ClassifierLayer(len_features, output_sizes[-1], num_of_layers=num_fc_layers, repeatInput=repeatInput, relu_last_layer=relu_last_layer),
    }
    
    use_multiclass = (len(output_sizes)>1)

    for indx, i in enumerate(phylo_distances):
        level_name = str(i).replace(".", "")+"distance"
        
        out_size = output_sizes[0 if not use_multiclass else indx]

        classification_layers[level_name] = ClassifierLayer(
                int((indx+1)*len_features/n_phylolevels), 
                out_size, 
                num_of_layers=num_fc_layers, repeatInput=repeatInput, relu_last_layer=relu_last_layer
            )

    return torch.nn.ModuleDict(classification_layers)

###----------------------####

class Reshape(nn.Module):
    def __init__(self, embed_dim, resolution, codes_perlevel_perkernel, levels=1, reverse=False):
        super(Reshape, self).__init__()
        self.embed_dim = embed_dim
        self.resolution = resolution
        self.levels = levels
        self.codes_perlevel_perkernel = codes_perlevel_perkernel
        self.reverse = reverse
        
    #(b, e_dim, cperlevel*r*r, l) -> (b, cperlevel*l*e_dim, r, r)
    def reverse_forward(self, x):
        assert x.shape[1] == self.embed_dim
        assert x.shape[3] == self.levels
        
        x = x.view(x.shape[0], self.embed_dim, -1, self.resolution*self.resolution, self.levels)
        x = x.permute(0, 1, 4, 2, 3)
        x = x.view(x.shape[0], self.embed_dim, self.levels, -1, self.resolution, self.resolution)
        x = x.reshape((x.shape[0], -1, self.resolution, self.resolution))
        

        return x

    # (b, cperlevel*l*e_dim, r, r) -> (b, e_dim, cperlevel*r*r, l)
    def forward(self, x):
        if self.reverse:
            return self.reverse_forward(x)
        
        assert x.shape[1] == self.embed_dim*self.levels*self.codes_perlevel_perkernel
        assert x.shape[2] == self.resolution
        assert x.shape[3] == self.resolution
        
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.view(x.shape[0], self.embed_dim, self.levels, -1, self.resolution*self.resolution)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.view(x.shape[0], self.embed_dim, -1 , self.levels)
        
        return x

#TODO: IMPORTANT!!!
# I am abandoning this class for now since it seems there are too many codes to analyse as a result of it compared to MLP.
# While training this model works, its analysis in terms of histograms and code replacement is broken.
#FIXME: if we decide to come back to it, let's fix it.
class PhyloDisentanglerConv(torch.nn.Module):
    def __init__(self, 
                in_channels, ch, out_ch, resolution, ## same as ddconfigs for autoencoder
                embed_dim, n_embed, # same as codebook configs
                n_phylo_channels, n_phylolevels, attribute_codes_per_phylolevel_perkernel, # The dimensions for the phylo descriptors.
                lossconfig, 
                n_mlp_layers=1, nonattribute_codes_perkernel=None, use_multiclass=False,
                lossconfig_phylo=None, lossconfig_kernelorthogonality=None, lossconfig_anticlassification=None, verbose=False): 
        super().__init__()

        self.ch = ch
        self.n_phylo_channels = n_phylo_channels
        self.n_phylolevels = n_phylolevels
        self.attribute_codes_per_phylolevel_perkernel = attribute_codes_per_phylolevel_perkernel
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.nonattribute_codes_perkernel = nonattribute_codes_perkernel
        self.verbose = verbose
        self.resolution = resolution

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
        
        # phylo conv
        self.conv_phylo_in = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm((self.n_phylo_channels, resolution, resolution)),
            torch.nn.Conv2d(self.n_phylo_channels,
                            n_phylolevels*attribute_codes_per_phylolevel_perkernel*embed_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            nn.SiLU(),
        )
        self.conv_phylo_out = nn.Sequential(
            torch.nn.Conv2d(n_phylolevels*attribute_codes_per_phylolevel_perkernel*embed_dim,
                            self.n_phylo_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            nn.SiLU(),
        )
        
        # non-phylo conv
        self.conv_nonattr_in = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm((self.ch - self.n_phylo_channels, resolution, resolution)),
            torch.nn.Conv2d(self.ch - self.n_phylo_channels,
                            nonattribute_codes_perkernel*embed_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            nn.SiLU(),
        )
        self.conv_nonattr_out = nn.Sequential(
            torch.nn.Conv2d(nonattribute_codes_perkernel*embed_dim,
                            self.ch - self.n_phylo_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            nn.SiLU(),
        )
        
        # reshapers
        self.reshape_in_phylo = Reshape(embed_dim, resolution, attribute_codes_per_phylolevel_perkernel, n_phylolevels)
        self.reshape_out_phylo = Reshape(embed_dim, resolution, attribute_codes_per_phylolevel_perkernel, n_phylolevels, reverse=True)
        self.reshape_in_nonattr = Reshape(embed_dim, resolution, nonattribute_codes_perkernel)
        self.reshape_out_nonattr = Reshape(embed_dim, resolution, nonattribute_codes_perkernel, reverse=True)
        

        # quantizer
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.embedding_converter = Embedding_Code_converter(self.quantize.get_codebook_entry_index, self.quantize.embedding, (self.embed_dim, attribute_codes_per_phylolevel_perkernel*resolution*resolution, self.n_phylolevels))


        # upsampling
        self.conv_out = nn.Sequential(
            nn.SiLU(),
            torch.nn.Conv2d(self.ch,
                            out_ch,
                            kernel_size=1,
                            stride=1,
                            padding=0),
        )

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
            len_features = n_phylolevels*attribute_codes_per_phylolevel_perkernel*embed_dim*resolution*resolution
            assert n_phylolevels==len(self.loss_phylo.phylo_distances)+1, "Number of phylo distances should be consistent in the settings."
            
            # Create classification layers.
            self.classification_layers = create_phylo_classifier_layers(len_features, self.phylo_disentangler.loss_phylo.classifier_output_sizes[-1], num_fc_layers, n_phylolevels, self.loss_phylo.phylo_distances, repeatInput=False, relu_last_layer=relu_last_layer)
            
            
            
        # Create anti-classification
        self.loss_anticlassification = None
        if lossconfig_anticlassification is not None:
            lossconfig_anticlassification['params'] = {**lossconfig_anticlassification['params'], **{'verbose': verbose}}
            self.loss_anticlassification= instantiate_from_config(lossconfig_anticlassification)
            assert self.ch - self.n_phylo_channels == self.n_phylo_channels, "Channels need to be split in half between phylo and nonphylo"

            # TODO: let's use same quantizer for now. Maybe we need a different one later.
            # self.anti_classification_layers = VectorQuantizer(n_embed, embed_dim, beta=0.25) #TODO: I think these are enough codes for now
            # self.codebook_mapping_layers = make_MLP([embed_dim, codebooks_per_phylolevel, n_levels_non_attribute], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=False)
            self.codebook_mapping_layers = nn.Sequential(
                torch.nn.Conv2d(nonattribute_codes_perkernel*embed_dim,
                                n_phylolevels*attribute_codes_per_phylolevel_perkernel*embed_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0),
                nn.SiLU(),
            )


        # print model
        print('phylovqgan', self)
        summary(self.cuda(), (1, in_channels, resolution, resolution))
    

    
    # NOTE: This does not return losses. Only used for outputting!
    def from_quant_only(self, quant_attribute, quant_nonattribute=None):        
        hout_phylo = self.conv_phylo_out(quant_attribute)
        
        if self.loss_anticlassification is not None:
            assert quant_nonattribute is not None, "Should have quant_nonattribute when using anticlass"
            hout_non_phylo = self.conv_nonattr_out(quant_nonattribute)
            h_ = torch.cat((hout_phylo, hout_non_phylo), 1)
            
        output = self.conv_out(h_)
        
        outputs = {CONSTANTS.DISENTANGLER_DECODER_OUTPUT: output}
        
        return outputs, {}
    
    def forward(self, input, overriding_quant=None):
        h = self.conv_in(input)

        # print(h.shape, self.n_phylo_channels, self.ch)
        h_phylo, h_img = torch.split(h, [self.n_phylo_channels, self.ch - self.n_phylo_channels], dim=1)
        z_phylo = self.conv_phylo_in(h_phylo)
        zq_phylo, q_phylo_loss, info = self.quantize(self.reshape_in_phylo(z_phylo))

        if overriding_quant is not None:
            assert zq_phylo.shape == overriding_quant.shape, str(zq_phylo.shape) + "!=" + str(overriding_quant.shape)
            zq_phylo = overriding_quant#torch.zeros_like(zq_phylo) #overriding_quant
        zq_phylo_reshaped = self.reshape_out_phylo(zq_phylo)
        hout_phylo = self.conv_phylo_out(zq_phylo_reshaped)
        
        loss_dic = {'quantizer_loss': q_phylo_loss}
        outputs = {CONSTANTS.QUANTIZED_PHYLO_OUTPUT: zq_phylo}

        

        if self.loss_anticlassification is not None:
            z_nonphylo = self.conv_nonattr_in(h_img)
            zq_nonphylo, q_nonphylo_loss, _ = self.quantize(self.reshape_in_nonattr(z_nonphylo))
            loss_dic = {'quantizer_loss': q_phylo_loss + q_nonphylo_loss}
            
            zq_nonphylo_reshaped = self.reshape_out_nonattr(zq_nonphylo)
            hout_non_phylo = self.conv_nonattr_out(zq_nonphylo_reshaped)
            h_ = torch.cat((hout_phylo, hout_non_phylo), 1)
            
            mapping_loss, learning_loss = self.loss_anticlassification(self.codebook_mapping_layers, zq_nonphylo_reshaped, zq_phylo_reshaped)
        
            loss_dic['anti_classification_mapping_loss'] = mapping_loss
            loss_dic['anti_classification_learning_loss'] = learning_loss
        
            if self.loss_phylo is not None:
                outputs[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT] = self.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](self.codebook_mapping_layers(zq_nonphylo_reshaped))
                
        else:
            h_ = torch.cat((hout_phylo, h_img), 1)

        output = self.conv_out(h_)
        
        outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]= output


        if self.loss_anticlassification is not None:
            outputs[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] = zq_nonphylo
            

        # Phylo networks
        if self.loss_phylo is not None:
            for name, layer in self.classification_layers.items():
                num_of_levels_included = int(layer.get_inputsize()/(self.embed_dim * self.attribute_codes_per_phylolevel_perkernel * self.resolution * self.resolution))
                outputs[name] = layer(zq_phylo[:, :, :, :num_of_levels_included]) # 0 for level 1, 0:1 for level 2, etc.

        if self.loss_kernelorthogonality is not None:
            kernel_orthogonality_loss = self.loss_kernelorthogonality(self.conv_in[1].weight)
            loss_dic['kernel_orthogonality_loss'] = kernel_orthogonality_loss
            
    
        return outputs, loss_dic   




###----------------------####


class PhyloDisentangler(torch.nn.Module):
    def __init__(self, 
                in_channels, ch, out_ch, resolution, ## same ad ddconfigs for autoencoder
                embed_dim, n_embed, # same as codebook configs
                n_phylo_channels, n_phylolevels, codebooks_per_phylolevel, # The dimensions for the phylo descriptors.
                lossconfig, 
                n_mlp_layers=1, n_levels_non_attribute=None, relu_last_layer=True, repeatInput=True, convin_switch=False,
                lossconfig_phylo=None, lossconfig_kernelorthogonality=None, lossconfig_anticlassification=None, verbose=False): 
        super().__init__()

        self.ch = ch
        self.n_phylo_channels = n_phylo_channels
        self.n_phylolevels = n_phylolevels
        self.codebooks_per_phylolevel = codebooks_per_phylolevel
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.n_levels_non_attribute = n_levels_non_attribute
        self.convin_switch = convin_switch
        
        self.passthrough = (self.ch != self.n_phylo_channels)

        self.verbose = verbose

        self.loss = instantiate_from_config(lossconfig)

        # downsampling
        if not convin_switch:
            self.conv_in = nn.Sequential(
                nn.SiLU(),
                torch.nn.Conv2d(in_channels,
                                self.ch,
                                kernel_size=1,
                                stride=1,
                                padding=0),
            )
        else:
            self.conv_in = nn.Sequential(
                torch.nn.Conv2d(in_channels,
                                self.ch,
                                kernel_size=1,
                                stride=1,
                                padding=0),
                nn.SiLU(),
            )

        # phylo MLP
        self.mlp_in = make_MLP([self.n_phylo_channels,resolution,resolution], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=True)
        self.mlp_out = make_MLP([embed_dim, codebooks_per_phylolevel, n_phylolevels], [self.n_phylo_channels,resolution,resolution], n_mlp_layers, normalize=False)
        
        self.mlp_in_non_attribute = None
        self.mlp_out_non_attribute = None
        if n_levels_non_attribute is not None:
            self.mlp_in_non_attribute = make_MLP([self.ch - self.n_phylo_channels,resolution,resolution], [embed_dim, codebooks_per_phylolevel, n_levels_non_attribute], n_mlp_layers, normalize=True)
            self.mlp_out_non_attribute = make_MLP([embed_dim, codebooks_per_phylolevel, n_levels_non_attribute], [self.ch - self.n_phylo_channels,resolution,resolution], n_mlp_layers, normalize=False)
            

        # quantizer
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.embedding_converter = Embedding_Code_converter(self.quantize.get_codebook_entry_index, self.quantize.embedding, (self.embed_dim, self.codebooks_per_phylolevel, self.n_phylolevels))


        # upsampling
        self.conv_out = nn.Sequential(
            nn.SiLU(),
            torch.nn.Conv2d(self.ch,
                            out_ch,
                            kernel_size=1,
                            stride=1,
                            padding=0),
        )

        self.loss_kernelorthogonality = None
        if lossconfig_kernelorthogonality is not None:
            lossconfig_kernelorthogonality['params'] = {**lossconfig_kernelorthogonality['params'], **{'verbose': verbose}}
            self.loss_kernelorthogonality= instantiate_from_config(lossconfig_kernelorthogonality)



        self.loss_phylo = None
        if lossconfig_phylo is not None:
            
            #TODO: maybe move logic here onward into its own function.

            # get loss and parse params
            lossconfig_phylo['params'] = {**lossconfig_phylo['params'], **{'verbose': verbose}}
            self.loss_phylo = instantiate_from_config(lossconfig_phylo)
            num_fc_layers = self.loss_phylo.fc_layers
            len_features = n_phylolevels*codebooks_per_phylolevel*embed_dim
            assert n_phylolevels==len(self.loss_phylo.phylo_distances)+1, "Number of phylo distances should be consistent in the settings."
            
            # Create classification layers.
            self.classification_layers = create_phylo_classifier_layers(len_features, self.loss_phylo.classifier_output_sizes, num_fc_layers, n_phylolevels, self.loss_phylo.phylo_distances, relu_last_layer=relu_last_layer, repeatInput=repeatInput)
            
            
            
        # Create anti-classification
        self.loss_anticlassification = None
        assert (self.passthrough) or (lossconfig_anticlassification is None), "Can't have anti-classification codebook without passthrough"
        if lossconfig_anticlassification is not None:
            lossconfig_anticlassification['params'] = {**lossconfig_anticlassification['params'], **{'verbose': verbose}}
            self.loss_anticlassification= instantiate_from_config(lossconfig_anticlassification)
            if n_levels_non_attribute is None:
                assert self.ch - self.n_phylo_channels == self.n_phylo_channels, "Channels need to be split in half between phylo and nonphylo"

            # TODO: let's use same quantizer for now. Maybe we need a different one later.
            # self.anti_classification_layers = VectorQuantizer(n_embed, embed_dim, beta=0.25) #TODO: I think these are enough codes for now
            if n_levels_non_attribute is None:
                self.codebook_mapping_layers = make_MLP([embed_dim, codebooks_per_phylolevel, n_phylolevels], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=False)
            else:
                self.codebook_mapping_layers = make_MLP([embed_dim, codebooks_per_phylolevel, n_levels_non_attribute], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=False)


        # print model
        print('phylovqgan', self)
        summary(self.cuda(), (1, in_channels, resolution, resolution))
    
    
    # NOTE: This does not return losses. Only used for outputting!
    def from_quant_only(self, quant_attribute, quant_nonattribute=None):
        assert (not self.passthrough) or (self.loss_anticlassification is not None), "Cannot be done for passthrough with no anticlass"
        
        hout_phylo = self.mlp_out(quant_attribute)
        if self.passthrough:
            if self.loss_anticlassification is not None:
                assert quant_nonattribute is not None, "Should have quant_nonattribute when using anticlass"
                if self.n_levels_non_attribute is not None:
                    hout_non_phylo = self.mlp_out_non_attribute(quant_nonattribute)
                else:
                    hout_non_phylo = self.mlp_out(quant_nonattribute)
                h_ = torch.cat((hout_phylo, hout_non_phylo), 1)
        else:
            h_ = hout_phylo
            
        output = self.conv_out(h_)
        
        outputs = {CONSTANTS.DISENTANGLER_DECODER_OUTPUT: output}
        
        return outputs, {}
    
    def encode(self, input, overriding_quant=None):
        h = self.conv_in(input)

        # print(h.shape, self.n_phylo_channels, self.ch)
        if self.passthrough:
            h_phylo, h_img = torch.split(h, [self.n_phylo_channels, self.ch - self.n_phylo_channels], dim=1)
        else:
            h_img = None
            h_phylo = h
        z_phylo = self.mlp_in(h_phylo)
        zq_phylo, q_phylo_loss, info_attr = self.quantize(z_phylo)

        if overriding_quant is not None:
            assert zq_phylo.shape == overriding_quant.shape, str(zq_phylo.shape) + "!=" + str(overriding_quant.shape)
            zq_phylo = overriding_quant#torch.zeros_like(zq_phylo) #overriding_quant
        
        
        loss_dic = {'quantizer_loss': q_phylo_loss}
        outputs = {CONSTANTS.QUANTIZED_PHYLO_OUTPUT: zq_phylo}

        zq_nonphylo = None
        info_nonattr = None
        if self.passthrough:
            if self.loss_anticlassification is not None:
                if self.n_levels_non_attribute is not None:
                    z_nonphylo = self.mlp_in_non_attribute(h_img)
                else:
                    z_nonphylo = self.mlp_in(h_img)
                zq_nonphylo, q_nonphylo_loss, info_nonattr = self.quantize(z_nonphylo)
                loss_dic = {'quantizer_loss': q_phylo_loss + q_nonphylo_loss}
                
        if self.loss_kernelorthogonality is not None:
            kernel_orthogonality_loss = self.loss_kernelorthogonality(self.conv_in[1 if not self.convin_switch else 0].weight)
            loss_dic['kernel_orthogonality_loss'] = kernel_orthogonality_loss    
                            
        return zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, info_attr, info_nonattr
    
    def decode(self, zq_phylo, zq_nonphylo, loss_dic={}, outputs={}, h_img=None):
        hout_phylo = self.mlp_out(zq_phylo)
        
        if self.passthrough:
            if self.loss_anticlassification is not None:
                
                if self.n_levels_non_attribute is not None:
                    hout_non_phylo = self.mlp_out_non_attribute(zq_nonphylo)
                else:    
                    hout_non_phylo = self.mlp_out(zq_nonphylo)
                h_ = torch.cat((hout_phylo, hout_non_phylo), 1)
                
                mapping_loss, learning_loss = self.loss_anticlassification(self.codebook_mapping_layers, zq_nonphylo, zq_phylo)
            
                loss_dic['anti_classification_mapping_loss'] = mapping_loss
                loss_dic['anti_classification_learning_loss'] = learning_loss
            
                if self.loss_phylo is not None:
                    outputs[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT] = self.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](self.codebook_mapping_layers(zq_nonphylo))
                    
            else:
                if h_img is None:
                    raise "Need the nonattribute convolutional features when there is passthrough with no anticlassififcation"
                h_ = torch.cat((hout_phylo, h_img), 1)
        else:
            h_ = hout_phylo

        output = self.conv_out(h_)
        
        outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]= output

        if self.loss_anticlassification is not None:
            outputs[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] = zq_nonphylo
            

        # Phylo networks
        if self.loss_phylo is not None:
            for name, layer in self.classification_layers.items():
                num_of_levels_included = int(layer.get_inputsize()/(self.embed_dim * self.codebooks_per_phylolevel))
                outputs[name] = layer(zq_phylo[:, :, :, :num_of_levels_included]) # 0 for level 1, 0:1 for level 2, etc.
                
        return outputs, loss_dic


    
    def forward(self, input, overriding_quant=None):
        zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, _, _ = self.encode(input, overriding_quant)
        outputs, loss_dic = self.decode(zq_phylo, zq_nonphylo, loss_dic, outputs, h_img)
    
        return outputs, loss_dic   

#*********************************



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

        phylo_args = args[CONSTANTS.PHYLOCONFIG_KEY]
        del args[CONSTANTS.PHYLOCONFIG_KEY]
        
        disentangler_type = 'MLP'
        if CONSTANTS.DISENTANGLERTYPE_KEY in args:
            disentangler_type = args[CONSTANTS.DISENTANGLERTYPE_KEY]
            del args[CONSTANTS.DISENTANGLERTYPE_KEY]
            
        complete_ckpt_path = None
        if CONSTANTS.COMPLETE_CKPT_KEY in args:
            complete_ckpt_path = args[CONSTANTS.COMPLETE_CKPT_KEY]
            del args[CONSTANTS.COMPLETE_CKPT_KEY]

        super().__init__(**args)

        self.freeze()

        if disentangler_type == 'MLP':
            self.phylo_disentangler = PhyloDisentangler(**phylo_args)
        elif disentangler_type == 'CONV':
            self.phylo_disentangler = PhyloDisentanglerConv(**phylo_args)
        else:
            raise('Disentangler type ' + disentangler_type + ' is invalid')

        self.verbose = phylo_args.get('verbose', False)
        
        if complete_ckpt_path is not None:
            sd = torch.load(complete_ckpt_path, map_location="cpu")["state_dict"]
            self.load_state_dict(sd, strict=True)
        
        # print model
        # print('totalmodel', self)
        # summary(self.cuda(), (1, 3, 512, 512))

    def encode(self, x, overriding_quant=None):
        encoder_out = self.encoder(x)
        zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, info_attr, info_nonattr = self.phylo_disentangler.encode(encoder_out, overriding_quant)
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
    
    def forward(self, input, overriding_quant=None):
        zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, encoder_out, _, _ = self.encode(input, overriding_quant)
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
        
    
    #TODO: make this only for debugging.
    def forward_hypothetical(self, input):
        encoder_out = self.encoder(input)
        h = self.quant_conv(encoder_out)
        quant, base_hypothetical_quantizer_loss, info = self.quantize(h)
        dec, _, _, _ = self.decode(quant)
        return dec, base_hypothetical_quantizer_loss

    def step(self, batch, batch_idx, prefix):
        x = self.get_input(batch, self.image_key)
        xrec, disentangler_loss_dic, base_loss_dic, in_out_disentangler = self(x)
        out_class_disentangler = {i:in_out_disentangler[i] for i in in_out_disentangler if i not in CONSTANTS.NON_CLASS_TENSORS}

        if self.verbose:
            xrec_hypthetical, base_hypothetical_quantizer_loss = self.forward_hypothetical(x)
            hypothetical_rec_loss =torch.mean(torch.abs(x.contiguous() - xrec_hypthetical.contiguous()))
            self.log(prefix+"/base_hypothetical_rec_loss", hypothetical_rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            self.log(prefix+"/base_hypothetical_quantizer_loss", base_hypothetical_quantizer_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        # base losses
        true_rec_loss = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        self.log(prefix+"/base_true_rec_loss", true_rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log(prefix+"/base_quantizer_loss", base_loss_dic['quantizer_loss'], prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # autoencode
        quantizer_disentangler_loss = disentangler_loss_dic['quantizer_loss']
        total_loss, log_dict_ae = self.phylo_disentangler.loss(quantizer_disentangler_loss, in_out_disentangler[CONSTANTS.DISENTANGLER_ENCODER_INPUT], in_out_disentangler[CONSTANTS.DISENTANGLER_DECODER_OUTPUT], 0, self.global_step, split=prefix)



        if self.phylo_disentangler.loss_kernelorthogonality is not None:
            kernelorthogonality_disentangler_loss = disentangler_loss_dic['kernel_orthogonality_loss']
            total_loss = total_loss + kernelorthogonality_disentangler_loss*self.phylo_disentangler.loss_kernelorthogonality.weight
            self.log(prefix+"/disentangler_kernelorthogonality_loss", kernelorthogonality_disentangler_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        


        if self.phylo_disentangler.loss_phylo is not None:
            phylo_losses_dict = self.phylo_disentangler.loss_phylo(total_loss, out_class_disentangler, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
            total_loss = phylo_losses_dict['cumulative_loss']

            self.log(prefix+CONSTANTS.DISENTANGLER_PHYLO_LOSS, phylo_losses_dict['total_phylo_loss'], prog_bar=self.verbose, logger=True, on_step=self.verbose, on_epoch=True)
            
        if self.phylo_disentangler.loss_anticlassification:
            mapping_loss = disentangler_loss_dic['anti_classification_mapping_loss']
            learning_loss = disentangler_loss_dic['anti_classification_learning_loss']
            total_loss = total_loss + (mapping_loss*self.phylo_disentangler.loss_anticlassification.beta + learning_loss)*self.phylo_disentangler.loss_anticlassification.weight
            self.log(prefix+"/disentangler_anti_classification_loss", mapping_loss, prog_bar=False, logger=True, on_step=self.verbose, on_epoch=True)
            if self.phylo_disentangler.loss_phylo:
                anti_classification_classifier_output = in_out_disentangler[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT]
                anti_classification_f1_score = self.phylo_disentangler.loss_phylo.F1(anti_classification_classifier_output, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                self.log(prefix+"/anti_classification_classifier_output", anti_classification_f1_score, prog_bar=False, logger=True, on_step=self.verbose, on_epoch=True)

        if self.verbose:
            with torch.no_grad():
                _, _, _, in_out_disentangler_of_rec = self(xrec)
                rec_classification = in_out_disentangler_of_rec[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
                generated_f1_score = self.phylo_disentangler.loss_phylo.F1(rec_classification, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                self.log(prefix+"/generated_f1_score", generated_f1_score, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        if self.verbose:
            self.log(prefix+"/disentangler_total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            for i in phylo_losses_dict['individual_losses']:
                self.log(prefix+"/disentangler_phylo_"+i, phylo_losses_dict['individual_losses'][i], prog_bar=True, logger=True, on_step=False, on_epoch=True)


        for i in phylo_losses_dict:
            if "_f1" in i:
                self.log(prefix+"/disentangler_phylo_"+i, phylo_losses_dict[i], prog_bar=True, logger=True, on_step=False, on_epoch=True)


        rec_loss = log_dict_ae[prefix+"/rec_loss"]
        self.log(prefix+"/disentangler_quantizer_loss", quantizer_disentangler_loss, prog_bar=True, logger=True, on_step=self.verbose, on_epoch=True)
        self.log(prefix+"/disentangler_rec_loss", rec_loss, prog_bar=True, logger=self.verbose, on_step=self.verbose, on_epoch=True)        

        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return total_loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    ##################### test ###########
        
    # TODO: This is kinda hacky. But ok for now for test purposes.
    def set_test_chkpt_path(self, chkpt_path):
        self.test_chkpt_path = chkpt_path

    @torch.no_grad() #TODO: maybe put this for all analysis scritps?
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

        #TODO: add checks for test if there is loss_phylo or not.
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
        
        # lr_schedulers = {}
        # if self.lr_scheduler_metric is not None:
        #     lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ae), "monitor": self.lr_scheduler_metric}
        
        return [opt_ae], [] #lr_schedulers