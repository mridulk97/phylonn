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

from taming.analysis_utils import get_CosineDistance_matrix, get_species_phylo_distance, get_HammingDistance_matrix, Embedding_Code_converter
from taming.analysis_utils import get_zqphylo_sub
from taming.plotting_utils import plot_heatmap, dump_to_json

# from torchsummary import summary
from torchinfo import summary
import collections

import taming.constants as CONSTANTS
import itertools

from torchmetrics import F1Score

import math


TEST_DIR="results_summary"



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
        


def create_phylo_classifier_layers(len_features, output_size, num_fc_layers, n_phylolevels, phylo_distances):
    classification_layers = {
        CONSTANTS.DISENTANGLER_CLASS_OUTPUT: ClassifierLayer(len_features, output_size, num_of_layers=num_fc_layers),
    }

    for indx, i in enumerate(phylo_distances):
        level_name = str(i).replace(".", "")+"distance"

        classification_layers[level_name] = ClassifierLayer(
                int((indx+1)*len_features/n_phylolevels), 
                output_size, 
                num_of_layers=num_fc_layers
            )

    return torch.nn.ModuleDict(classification_layers)



###----------------------####


class PhyloDisentangler(torch.nn.Module):
    def __init__(self, 
                in_channels, ch, out_ch, resolution, ## same ad ddconfigs for autoencoder
                embed_dim, n_embed, # same as codebook configs
                n_phylo_channels, n_phylolevels, codebooks_per_phylolevel, # The dimensions for the phylo descriptors.
                lossconfig, 
                n_mlp_layers=1,
                lossconfig_phylo=None, lossconfig_kernelorthogonality=None, lossconfig_anticlassification=None, verbose=False): 
        super().__init__()

        self.ch = ch
        self.n_phylo_channels = n_phylo_channels
        self.n_phylolevels = n_phylolevels
        self.codebooks_per_phylolevel = codebooks_per_phylolevel
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.passthrough = (self.ch != self.n_phylo_channels)

        self.verbose = verbose

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
        self.mlp_in = make_MLP([self.n_phylo_channels,resolution,resolution], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=True)
        self.mlp_out = make_MLP([embed_dim, codebooks_per_phylolevel, n_phylolevels], [self.n_phylo_channels,resolution,resolution], n_mlp_layers, normalize=False)

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
            output_size = self.loss_phylo.classifier_output_size
            num_fc_layers = self.loss_phylo.fc_layers
            len_features = n_phylolevels*codebooks_per_phylolevel*embed_dim
            assert n_phylolevels==len(self.loss_phylo.phylo_distances)+1, "Number of phylo distances should be consistent in the settings."
            
            # Create classification layers.
            self.classification_layers = create_phylo_classifier_layers(len_features, output_size, num_fc_layers, n_phylolevels, self.loss_phylo.phylo_distances)
            
            
            
        # Create anti-classification
        self.loss_anticlassification = None
        assert (self.passthrough) or (lossconfig_anticlassification is None), "Can't have anti-classification codebook without passthrough"
        if lossconfig_anticlassification is not None:
            lossconfig_anticlassification['params'] = {**lossconfig_anticlassification['params'], **{'verbose': verbose}}
            self.loss_anticlassification= instantiate_from_config(lossconfig_anticlassification)
            assert self.ch - self.n_phylo_channels == self.n_phylo_channels, "Channels need to be split in half between phylo and nonphylo"

            # TODO: let's use same quantizer for now. Maybe we need a different one later.
            # self.anti_classification_layers = VectorQuantizer(n_embed, embed_dim, beta=0.25) #TODO: I think these are enough codes for now
            #TOOD: maybe we need more layers?
            self.codebook_mapping_layers = make_MLP([embed_dim, codebooks_per_phylolevel, n_phylolevels], [embed_dim, codebooks_per_phylolevel, n_phylolevels], n_mlp_layers, normalize=False)


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
                h_ = torch.cat((hout_phylo, self.mlp_out(quant_nonattribute)), 1)
        else:
            h_ = hout_phylo
            
        output = self.conv_out(h_)
        
        outputs = {CONSTANTS.DISENTANGLER_DECODER_OUTPUT: output}
        
        return outputs, {}
    
    def forward(self, input, overriding_quant=None):
        h = self.conv_in(input)

        # print(h.shape, self.n_phylo_channels, self.ch)
        if self.passthrough:
            h_phylo, h_img = torch.split(h, [self.n_phylo_channels, self.ch - self.n_phylo_channels], dim=1)
        else:
            h_phylo = h
        z_phylo = self.mlp_in(h_phylo)
        zq_phylo, q_phylo_loss, info = self.quantize(z_phylo)

        if overriding_quant is not None:
            assert zq_phylo.shape == overriding_quant.shape, str(zq_phylo.shape) + "!=" + str(overriding_quant.shape)
            zq_phylo = overriding_quant#torch.zeros_like(zq_phylo) #overriding_quant
        hout_phylo = self.mlp_out(zq_phylo)
        
        loss_dic = {'quantizer_loss': q_phylo_loss}
        outputs = {CONSTANTS.QUANTIZED_PHYLO_OUTPUT: zq_phylo}

        
        if self.passthrough:
            if self.loss_anticlassification is not None:
                z_nonphylo = self.mlp_in(h_img)
                zq_nonphylo, q_nonphylo_loss, _ = self.quantize(z_nonphylo)
                loss_dic = {'quantizer_loss': q_phylo_loss + q_nonphylo_loss}
                h_ = torch.cat((hout_phylo, self.mlp_out(zq_nonphylo)), 1)
                
                mapping_loss, learning_loss = self.loss_anticlassification(self.codebook_mapping_layers, zq_nonphylo, zq_phylo)
            
                loss_dic['anti_classification_mapping_loss'] = mapping_loss
                loss_dic['anti_classification_learning_loss'] = learning_loss
            
                if self.verbose:
                    if self.loss_phylo is not None:
                        outputs[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT] = self.classification_layers[CONSTANTS.DISENTANGLER_CLASS_OUTPUT](self.codebook_mapping_layers(zq_nonphylo))
                    
            else:
                h_ = torch.cat((hout_phylo, h_img), 1)
        else:
            h_ = hout_phylo

        output = self.conv_out(h_)
        
        outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]= output


        if self.passthrough and self.loss_anticlassification is not None:
            outputs[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] = zq_nonphylo
            

        # Phylo networks
        if self.loss_phylo is not None:
            for name, layer in self.classification_layers.items():
                num_of_levels_included = int(layer.get_inputsize()/(self.embed_dim * self.codebooks_per_phylolevel))
                outputs[name] = layer(zq_phylo[:, :, :, :num_of_levels_included]) # 0 for level 1, 0:1 for level 2, etc.

        if self.loss_kernelorthogonality is not None:
            kernel_orthogonality_loss = self.loss_kernelorthogonality(self.conv_in[1].weight)
            loss_dic['kernel_orthogonality_loss'] = kernel_orthogonality_loss
            
    
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

        phylo_args = args[CONSTANTS.PHYLOCONFIG_KEY]
        del args[CONSTANTS.PHYLOCONFIG_KEY]

        super().__init__(**args)

        self.freeze()

        self.phylo_disentangler = PhyloDisentangler(**phylo_args)

        self.verbose = phylo_args.get('verbose', False)

        # print model
        # print('totalmodel', self)
        # summary(self.cuda(), (1, 3, 512, 512))

    def encode(self, x, overriding_quant=None):
        encoder_out = self.encoder(x)
        #phylo_quantizer_loss, classification_phylo_loss
        disentangler_outputs, disentangler_loss_dic = self.phylo_disentangler(encoder_out, overriding_quant)
        disentangler_out = disentangler_outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]
        h = self.quant_conv(disentangler_out)
        quant, base_quantizer_loss, info = self.quantize(h)

        #consolidate dicts
        base_loss_dic = {'quantizer_loss': base_quantizer_loss}
        in_out_disentangler = {
            CONSTANTS.DISENTANGLER_ENCODER_INPUT: encoder_out,
        }
        in_out_disentangler = {**in_out_disentangler, **disentangler_outputs}

        return quant, disentangler_loss_dic, base_loss_dic, in_out_disentangler, info

    def forward(self, input, overriding_quant=None):
        quant, disentangler_loss_dic, base_loss_dic, in_out_disentangler, _ = self.encode(input, overriding_quant)
        dec = self.decode(quant)
        return dec, disentangler_loss_dic, base_loss_dic, in_out_disentangler
    
    #NOTE: This does not return losses. Only used for outputting!
    def from_quant_only(self, quant, quant_nonattribute=None):
        disentangler_outputs, _ = self.phylo_disentangler.from_quant_only(quant, quant_nonattribute)
        disentangler_out = disentangler_outputs[CONSTANTS.DISENTANGLER_DECODER_OUTPUT]
        h = self.quant_conv(disentangler_out)
        quant, _, _ = self.quantize(h)
        dec = self.decode(quant)
        return dec, {}
        
    
    #TODO: make this only for debugging.
    def forward_hypothetical(self, input):
        encoder_out = self.encoder(input)
        h = self.quant_conv(encoder_out)
        quant, base_hypothetical_quantizer_loss, info = self.quantize(h)
        dec = self.decode(quant)
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

            self.log(prefix+"/disentangler_phylo_loss", phylo_losses_dict['total_phylo_loss'], prog_bar=self.verbose, logger=True, on_step=self.verbose, on_epoch=True)
            
        if self.phylo_disentangler.loss_anticlassification:
            mapping_loss = disentangler_loss_dic['anti_classification_mapping_loss']
            learning_loss = disentangler_loss_dic['anti_classification_learning_loss']
            total_loss = total_loss + (mapping_loss*self.phylo_disentangler.loss_anticlassification.beta + learning_loss)*self.phylo_disentangler.loss_anticlassification.weight
            self.log(prefix+"/disentangler_anti_classification_loss", mapping_loss, prog_bar=False, logger=True, on_step=self.verbose, on_epoch=True)
            if self.verbose and self.phylo_disentangler.loss_phylo:
                anti_classification_classifier_output = in_out_disentangler[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_CLASS_OUTPUT]
                anti_classification_f1_score = self.phylo_disentangler.loss_phylo.F1(anti_classification_classifier_output, batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])
                self.log(prefix+"/anti_classification_classifier_output", anti_classification_f1_score, prog_bar=False, logger=True, on_step=self.verbose, on_epoch=True)

        if self.verbose:
            self.log(prefix+"/disentangler_total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
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


    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    ##################### test ###########
        
    # TODO: This is kinda hacky. But ok for now for test purposes.
    def set_test_chkpt_path(self, chkpt_path):
        self.test_chkpt_path = chkpt_path

    def test_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        _, _, _, in_out_disentangler = self(x)

        pred_ = in_out_disentangler[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
        class_ = batch[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
        classname_ = batch[CONSTANTS.DATASET_CLASSNAME]
        zq_phylo_features = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
        
        return {
            'pred': pred_, 
            CONSTANTS.DISENTANGLER_CLASS_OUTPUT: class_, #.unsqueeze(-1)
            CONSTANTS.DATASET_CLASSNAME: classname_,
            'zq_phylo': zq_phylo_features
        }
    
    def test_epoch_end(self, in_out):
        test_measures = {}

        preds =torch.cat([x['pred'] for x in in_out], 0)
        classes = torch.cat([x[CONSTANTS.DISENTANGLER_CLASS_OUTPUT] for x in in_out], 0)
        classnames = list(itertools.chain.from_iterable([x[CONSTANTS.DATASET_CLASSNAME] for x in in_out]))
        zq_phylos = torch.cat([x['zq_phylo'] for x in in_out], 0)
        sorting_indices = numpy.argsort(classes.cpu())
        sorted_zq_phylos = zq_phylos[sorting_indices, :]
        sorted_zq_phylos_codes = self.phylo_disentangler.embedding_converter.get_phylo_codes(sorted_zq_phylos)
        reverse_shaped_sorted_zq_phylos_codes = self.phylo_disentangler.embedding_converter.reshape_code(sorted_zq_phylos_codes, reverse=True)
        soted_class_names = [classnames[i] for i in sorting_indices]
        sorted_unique_classes = sorted(set(soted_class_names))

        #TODO: add checks for test if there is loss_phylo or not.
        F1 = F1Score(num_classes=self.phylo_disentangler.loss_phylo.classifier_output_size, multiclass=True).to(preds.device)

        test_measures['class_f1'] = F1(preds, classes).item()

        
        num_of_levels = self.phylo_disentangler.n_phylolevels
        for level in range(num_of_levels):
        
            #****************

            # plots per specimen
            print("Calculating embedding distances for level {}".format(level))
            
            sub_sorted_zq_phylos_codes = self.phylo_disentangler.embedding_converter.reshape_code(reverse_shaped_sorted_zq_phylos_codes[:, :, :level+1])
            zq_hamming_distances = get_HammingDistance_matrix(sub_sorted_zq_phylos_codes)
            
            plot_heatmap(zq_hamming_distances.cpu(), self.test_chkpt_path, title='zq hamming distances for level {}'.format(level), postfix=TEST_DIR)
            
            #********************
            print("Calculating phylo distances for level {}".format(level))
            
            level_relative_distance = self.phylo_disentangler.loss_phylo.get_relative_distance_for_level(level)
            species_distances = get_species_phylo_distance(soted_class_names, self.phylo_disentangler.loss_phylo.phylogeny.get_distance_between_parents, relative_distance=level_relative_distance)
            plot_heatmap(species_distances, self.test_chkpt_path, title='phylo distances for level {}'.format(level), postfix=TEST_DIR)

            dump_to_json(test_measures, self.test_chkpt_path)
            
            
            #******************
            print("Calculating aggregated distances for level {}".format(level))
            # plot per species

            phylo_dist = torch.zeros(len(sorted_unique_classes), len(sorted_unique_classes))
            embedding_dist = torch.zeros(len(sorted_unique_classes), len(sorted_unique_classes))
            for indx_i, i in enumerate(sorted_unique_classes):
                class_i_indices = [idx for idx, element in enumerate(soted_class_names) if element == i] #numpy.where(sorted_classes == i)[0]
                for indx_j, j in enumerate(sorted_unique_classes[indx_i:]):
                    class_j_indices = [idx for idx, element in enumerate(soted_class_names) if element == j] # numpy.where(sorted_classes == j)[0] 
                    i_j_mean_embeddign_distance = torch.mean(zq_hamming_distances[class_i_indices, :][:, class_j_indices])
                    i_j_mean_phylo_distance = torch.mean(species_distances[class_i_indices, :][:, class_j_indices])
                    embedding_dist[indx_i][indx_j+ indx_i] = embedding_dist[indx_j+ indx_i][indx_i] = i_j_mean_embeddign_distance
                    phylo_dist[indx_i][indx_j+ indx_i] = phylo_dist[indx_j+ indx_i][indx_i] = i_j_mean_phylo_distance
            plot_heatmap(phylo_dist.cpu(), self.test_chkpt_path, title='phylo species distances for level {}'.format(level), postfix=TEST_DIR)
            plot_heatmap(embedding_dist.cpu(), self.test_chkpt_path, title='zq hamming species distances for level {}'.format(level), postfix=TEST_DIR)

            

        return test_measures
    #######################################

    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(self.phylo_disentangler.parameters(), lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae], []