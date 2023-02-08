from scripts.import_utils import instantiate_from_config
from scripts.modules.losses.phyloloss import get_loss_name
import torch
from torch import nn
import numpy

from scripts.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from scripts.models.vqgan import VQModel

from scripts.analysis_utils import get_species_phylo_distance, get_HammingDistance_matrix, Embedding_Code_converter, aggregate_metric_from_specimen_to_species
from scripts.plotting_utils import plot_heatmap, dump_to_json, plot_confusionmatrix

# from torchsummary import summary
from torchinfo import summary

import scripts.constants as CONSTANTS
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
    def __init__(self, num_of_inputs, num_of_outputs, num_of_layers = 1, normalize=False): # bnorm=False,
        super(ClassifierLayer, self).__init__()
        
        self.num_of_inputs = num_of_inputs
        
        out_sizes = get_hidden_layer_sizes(num_of_inputs, num_of_outputs, num_of_layers)

        l = [torch.nn.Flatten()] 
        
        if normalize:
            l.append(nn.LayerNorm(num_of_inputs))
    
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

#TODO: probably won't need this in the official submission. can delete later.
class AttnClassificationLayer(torch.nn.Module):
    def __init__(self, attnconfig_copy): # bnorm=False,
        super(AttnClassificationLayer, self).__init__()
        self.num_of_inputs = attnconfig_copy['params']['block_size']
        self.embd_d = attnconfig_copy['params']['n_embd']
        
        print('moe', self.num_of_inputs, self.embd_d)
        self.l = instantiate_from_config(attnconfig_copy)

    def get_inputsize(self):
        return self.num_of_inputs

    
    def forward(self, input):
        input = input.reshape(input.shape[0], input.shape[1], -1).permute(0,2,1) #TODO: maybe a nicer way to do this?
        print('input', input.shape)
        return self.l(input)

#TODO: probably won't need this in the official submission. can delete later.
def create_phylo_classifier_attnlayers(len_features, output_sizes, attnconfig, n_phylolevels, embed_dim, phylo_distances):
    attnconfig_copy = attnconfig.copy()
    attnconfig_copy['params']['block_size'] = int(len_features/embed_dim)
    attnconfig_copy['params']['n_classes'] = output_sizes[-1] 
    classification_layers = {
        CONSTANTS.DISENTANGLER_CLASS_OUTPUT: AttnClassificationLayer(attnconfig_copy),
    }
        
    for indx, i in enumerate(phylo_distances):
        level_name = get_loss_name(phylo_distances, indx)

        attnconfig_copy = attnconfig.copy()
        attnconfig_copy['params']['block_size'] = int((indx+1)*len_features/(n_phylolevels*embed_dim))
        attnconfig_copy['params']['n_classes'] = output_sizes[indx]
        classification_layers[level_name] = AttnClassificationLayer(attnconfig_copy)

    return torch.nn.ModuleDict(classification_layers)

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
                n_mlp_layers=1, n_levels_non_attribute=None, base_rec_loss_weight=0.0,
                lossconfig_phylo=None, lossconfig_kernelorthogonality=None, lossconfig_anticlassification=None, selfattentionclassifier_config=None, verbose=False): 
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
            len_features = n_phylolevels*codebooks_per_phylolevel*embed_dim
            assert n_phylolevels==len(self.loss_phylo.phylo_distances)+1, "Number of phylo distances should be consistent in the settings."
            
            # Create classification layers.
            self.selfattentionclassifier_config = selfattentionclassifier_config
            if selfattentionclassifier_config is None:
                num_fc_layers = self.loss_phylo.fc_layers
                self.classification_layers = create_phylo_classifier_layers(len_features, self.loss_phylo.classifier_output_sizes, num_fc_layers, n_phylolevels, self.loss_phylo.phylo_distances)
            else:
                self.classification_layers = create_phylo_classifier_attnlayers(len_features, self.loss_phylo.classifier_output_sizes, selfattentionclassifier_config, n_phylolevels, embed_dim, self.loss_phylo.phylo_distances)
                
            
   
        # Create anti-classification
        self.loss_anticlassification = None
        if lossconfig_anticlassification is not None:
            lossconfig_anticlassification['params'] = {**lossconfig_anticlassification['params'], **{'verbose': verbose}}
            self.loss_anticlassification= instantiate_from_config(lossconfig_anticlassification)
            self.codebook_mapping_layers = make_MLP([embed_dim, codebooks_per_phylolevel, n_levels_non_attribute], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=False)    

        self.base_rec_loss_weight = base_rec_loss_weight

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
    
    def decode(self, zq_phylo, zq_nonphylo, loss_dic={}, outputs={}, labels=None, h_img=None):
        hout_phylo = self.mlp_out(zq_phylo)
            
        hout_non_phylo = self.mlp_out_non_attribute(zq_nonphylo)
        h_ = torch.cat((hout_phylo, hout_non_phylo), 1)
        
        # if self.loss_anticlassification is not None:
        #     if isinstance(self.loss_anticlassification, AntiClassificationLoss2):
        #         if self.loss_phylo is None or labels is None:
        #             raise "Can't have AntiClassificationLoss2 without a classification layer and labels."
        #         mapping_loss, learning_loss = self.loss_anticlassification(self.codebook_mapping_layers, self.classification_layers, zq_nonphylo, zq_phylo)
        #     else:
        #         mapping_loss, learning_loss = self.loss_anticlassification(self.codebook_mapping_layers, labels, zq_nonphylo)
        
        #     loss_dic['anti_classification_mapping_loss'] = mapping_loss
        #     loss_dic['anti_classification_learning_loss'] = learning_loss      
        
                
        if self.loss_anticlassification is not None:
            mapping_output, learning_output = self.loss_anticlassification(self.codebook_mapping_layers, zq_nonphylo)#, zq_phylo)
        
            outputs[CONSTANTS.DISENTANGLER_ANTI_MAPPING_OUTPUT] = mapping_output
            outputs[CONSTANTS.DISENTANGLER_ANTI_LEARNING_OUTPUT] = learning_output            

        output = self.conv_out(h_)
        
        outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]= output
        outputs[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] = zq_nonphylo
            

        # Phylo networks
        if self.loss_phylo is not None:
            for name, layer in self.classification_layers.items():
                num_of_levels_included = int(layer.get_inputsize()/(self.embed_dim * self.codebooks_per_phylolevel))
                o = zq_phylo[:, :, :, :num_of_levels_included]
                outputs[name] = layer(o) # 0 for level 1, 0:1 for level 2, etc.
        
        if self.loss_anticlassification is not None:
            o = self.codebook_mapping_layers(zq_nonphylo)
            outputs[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT] = self.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](o)
        return outputs, loss_dic


    
    def forward(self, input, overriding_quant_attr=None, overriding_quant_nonattr=None):
        zq_phylo, zq_nonphylo, loss_dic, outputs, h_img, _, _ = self.encode(input, overriding_quant_attr, overriding_quant_nonattr)
        outputs, loss_dic = self.decode(zq_phylo, zq_nonphylo, loss_dic, outputs, h_img)
    
        return outputs, loss_dic   

#*********************************



class PhyloVQVAE(VQModel):
    def __init__(self, **args):
        print(args)

        # For wandb
        self.save_hyperparameters()

        phylo_args = args[CONSTANTS.PHYLOCONFIG_KEY]
        del args[CONSTANTS.PHYLOCONFIG_KEY]
            
        self.lr_factor = args[CONSTANTS.LRFACTOR_KEY] if CONSTANTS.LRFACTOR_KEY in args.keys() else 0.01
        if CONSTANTS.LRFACTOR_KEY in args:
            del args[CONSTANTS.LRFACTOR_KEY]
            
        self.lr_cycle = args[CONSTANTS.LRCYCLE] if CONSTANTS.LRCYCLE in args.keys() else 150
        if CONSTANTS.LRCYCLE in args:
            del args[CONSTANTS.LRCYCLE]
            
        self.scheduler_type = args[CONSTANTS.SCHEDULER] if CONSTANTS.SCHEDULER in args.keys() else None
        if CONSTANTS.SCHEDULER in args:
            del args[CONSTANTS.SCHEDULER]
            
        self.optim_type = args[CONSTANTS.OPTIM] if CONSTANTS.OPTIM in args.keys() else None
        if CONSTANTS.OPTIM in args:
            del args[CONSTANTS.OPTIM]

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
            losses[prefix+ CONSTANTS.BASERECLOSS] = true_rec_loss
            losses[prefix+"/base_quantizer_loss"] = base_loss_dic['quantizer_loss']
            
            # autoencode
            quantizer_disentangler_loss = disentangler_loss_dic['quantizer_loss']
            total_loss, log_dict_ae = self.phylo_disentangler.loss(quantizer_disentangler_loss, in_out_disentangler[CONSTANTS.DISENTANGLER_ENCODER_INPUT], in_out_disentangler[CONSTANTS.DISENTANGLER_DECODER_OUTPUT], 0, self.global_step, split=prefix)


            if self.phylo_disentangler.base_rec_loss_weight > 0:      
                total_loss = total_loss + self.phylo_disentangler.base_rec_loss_weight*true_rec_loss      
            
            
            if self.phylo_disentangler.loss_kernelorthogonality is not None:
                kernelorthogonality_disentangler_loss = disentangler_loss_dic['kernel_orthogonality_loss']
                total_loss = total_loss + kernelorthogonality_disentangler_loss*self.phylo_disentangler.loss_kernelorthogonality.weight
                losses[prefix+"/disentangler_kernelorthogonality_loss"] = kernelorthogonality_disentangler_loss

            
            
            

            if self.phylo_disentangler.loss_phylo is not None:
                phylo_losses_dict = self.phylo_disentangler.loss_phylo(total_loss, out_class_disentangler, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                total_loss = phylo_losses_dict['cumulative_loss']
                losses[prefix+CONSTANTS.DISENTANGLER_PHYLO_LOSS] = phylo_losses_dict['total_phylo_loss']
                
                for i in phylo_losses_dict['individual_losses']:
                    losses[prefix+"/disentangler_phylo_"+i] = phylo_losses_dict['individual_losses'][i]

                for i in phylo_losses_dict:
                    if "_f1" in i:
                        losses[prefix+"/disentangler_phylo_"+i] = phylo_losses_dict[i]
            
            
            
            

            
            
            if self.phylo_disentangler.loss_anticlassification is not None:
                o = in_out_disentangler[CONSTANTS.DISENTANGLER_ANTI_LEARNING_OUTPUT]
                if self.phylo_disentangler.loss_anticlassification.v2==True:
                    class_learning = self.phylo_disentangler.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](o)
                    learning_loss = -torch.nn.CrossEntropyLoss()(class_learning, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                elif self.phylo_disentangler.loss_anticlassification.v3==True:
                    class_learning = self.phylo_disentangler.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](o)
                    class_learning = torch.nn.functional.softmax(class_learning, dim=1)
                    learning_loss = torch.mean(torch.sum(class_learning*class_learning.add(1e-9).log(), dim=1), dim=0) + numpy.log(class_learning.shape[1])
                else:
                    learning_loss = -torch.mean(torch.abs(in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT].detach().contiguous() - o.contiguous()))
                
                total_loss = total_loss + learning_loss*self.phylo_disentangler.loss_anticlassification.weight
                
                # if self.phylo_disentangler.loss_phylo:
                anti_classification_classifier_output = in_out_disentangler[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT]
                anti_classification_f1_score = self.phylo_disentangler.loss_phylo.F1(anti_classification_classifier_output, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                losses[prefix+"/anti_classification_classifier_output"] = anti_classification_f1_score





            # if self.verbose:
            with torch.no_grad():
                _, _, _, in_out_disentangler_of_rec = self(xrec)
                rec_classification = in_out_disentangler_of_rec[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
                generated_f1_score = self.phylo_disentangler.loss_phylo.F1(rec_classification, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                losses[prefix+"/generated_f1_score"] = generated_f1_score
            




            losses[prefix+"/disentangler_total_loss"] = total_loss




            rec_loss = log_dict_ae[prefix+"/rec_loss"]
            losses[prefix+"/disentangler_quantizer_loss"] = quantizer_disentangler_loss
            losses[prefix+"/disentangler_rec_loss"] = rec_loss
            
            self.log_dict(losses, logger=True, on_step=False, on_epoch=True)
            
            outputs = {
                'loss': total_loss,
                
                CONSTANTS.QUANTIZED_PHYLO_OUTPUT: in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT],
                CONSTANTS.DISENTANGLER_CLASS_OUTPUT: batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT],
                CONSTANTS.DATASET_CLASSNAME: batch[CONSTANTS.DATASET_CLASSNAME],
                'logs': losses
            }
            outputs[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT]
            
                
            return outputs
        
        if optimizer_idx==1 and (self.phylo_disentangler.loss_anticlassification is not None):
            o = in_out_disentangler[CONSTANTS.DISENTANGLER_ANTI_MAPPING_OUTPUT]
            if self.phylo_disentangler.loss_anticlassification.v2==True:
                class_mapping = self.phylo_disentangler.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](o)
                mapping_loss = torch.nn.CrossEntropyLoss()(class_mapping, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
            elif self.phylo_disentangler.loss_anticlassification.v3==True:
                class_mapping = self.phylo_disentangler.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](o)
                class_mapping = torch.nn.functional.softmax(class_mapping, dim=1)
                mapping_loss = -torch.mean(torch.sum(class_mapping*class_mapping.add(1e-9).log(), dim=1), dim=0)
            else:
                mapping_loss = torch.mean(torch.abs(in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT].detach().contiguous() - o.contiguous()))
            
            total_loss = mapping_loss*self.phylo_disentangler.loss_anticlassification.beta*self.phylo_disentangler.loss_anticlassification.weight
            losses = {prefix+"/disentangler_anti_classification_loss": mapping_loss}
            
            return {
                'loss': total_loss,
                'logs': losses
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
        
    @torch.no_grad()
    def on_validation_start(self):
        for v in self.trainer.val_dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

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
        zq_non_phylo_features = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT]
        
        _, _, _, in_out_disentangler_rec = self(rec_x)
        class_recreated = in_out_disentangler_rec[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
        
        return {
            'pred': pred_, 
            'pred_rec': class_recreated,
            CONSTANTS.DISENTANGLER_CLASS_OUTPUT: class_, #.unsqueeze(-1)
            CONSTANTS.DATASET_CLASSNAME: classname_,
            'zq_phylo': zq_phylo_features,
            'zq_nonphylo': zq_non_phylo_features
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
            
        #plots per specimen
        print("Calculating embedding distances for nonphylo")
        
        zq_nonphylo = torch.cat([x['zq_nonphylo'] for x in in_out], 0)
        sorting_indices = numpy.argsort(classes.cpu())
        sorted_zq_nonphylo = zq_nonphylo[sorting_indices, :]
        sorted_zq_nonphylo_codes = self.phylo_disentangler.embedding_converter.get_phylo_codes(zq_nonphylo)
        zq_hamming_distances = get_HammingDistance_matrix(sorted_zq_nonphylo_codes)
        
        plot_heatmap(zq_hamming_distances.cpu(), self.test_chkpt_path, title='zq hamming distances for nonphylo', postfix=CONSTANTS.TEST_DIR)
        
        #********************
        print("Calculating phylo distances for level {}".format(level))
        
        species_distances = get_species_phylo_distance(sorted_class_names_according_to_class_indx, self.phylo_disentangler.loss_phylo.phylogeny.get_distance_between_parents, relative_distance=1.0)
        plot_heatmap(species_distances, self.test_chkpt_path, title='phylo distances for nonphylo', postfix=CONSTANTS.TEST_DIR)
            
        #******************
        print("Calculating aggregated distances for level {}".format(level))
        
        embedding_dist = aggregate_metric_from_specimen_to_species(sorted_class_names_according_to_class_indx, zq_hamming_distances)
        plot_heatmap(embedding_dist.cpu(), self.test_chkpt_path, title='zq hamming species distances for nonphylo'.format(level), postfix=CONSTANTS.TEST_DIR)
            
        return test_measures
    #######################################

    
    def configure_optimizers(self):
        lr = self.learning_rate
        if self.optim_type is None:
            opt_ae = torch.optim.Adam(self.phylo_disentangler.parameters(), lr=lr, betas=(0.5, 0.9))
        else:
            opt_ae = torch.optim.AdamW(self.phylo_disentangler.parameters(), lr=lr, betas=(0.5, 0.9))
        opts = [opt_ae]
        if self.phylo_disentangler.loss_anticlassification is not None:
            if self.optim_type is None:
                opt_mapping = torch.optim.Adam(self.phylo_disentangler.codebook_mapping_layers.parameters(), lr=lr, betas=(0.5, 0.9))
            else:
                opt_mapping = torch.optim.AdamW(self.phylo_disentangler.codebook_mapping_layers.parameters(), lr=lr, betas=(0.5, 0.9))
            opts.append(opt_mapping)
        else:
            opts.append(opt_ae)
            
        
        if self.scheduler_type is None:
            lr_schedulers = [{
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_ae, self.lr_cycle, eta_min=lr*self.lr_factor),
                },{
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_ae, self.lr_cycle, eta_min=lr*self.lr_factor),
                },
            ]
        elif self.scheduler_type == "warmup":
            warm_up_step = 40
            def lr_foo(epoch):
                if epoch < warm_up_step:
                    # warm up lr
                    lr_scale = 0.1 ** ((warm_up_step - epoch)*0.01)
                else:
                    lr_scale = 0.97 ** epoch
                    if lr_scale < self.lr_factor:
                       lr_scale = self.lr_factor

                return lr_scale

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_ae,
                lr_lambda=lr_foo
            )
        
            lr_schedulers = [{
                "scheduler": scheduler,
                },{
                "scheduler": scheduler,
                },
            ]
        elif self.scheduler_type == "const":
            lr_schedulers = [
            ]
        else:
            raise "Undefined scheduler"
            
            
        return opts, lr_schedulers 