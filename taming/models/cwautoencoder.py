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
CONCEPT_DATA_KEY = "concept_data"

DISENTANGLER_DECODER_OUTPUT = 'output'
DISENTANGLER_ENCODER_INPUT = 'in'
DISENTANGLER_CLASS_OUTPUT = 'class'

class CWmodelVQGAN(VQModel):
    def __init__(self, **args):
        print(args)
        
        self.save_hyperparameters()

        # phylo_args = args[PHYLOCONFIG_KEY]
        # del args[PHYLOCONFIG_KEY]

        concept_data_args = args[CONCEPT_DATA_KEY]
        print("Concepts params : ", concept_data_args)
        self.concepts = instantiate_from_config(concept_data_args)
        self.concepts.prepare_data()
        self.concepts.setup()
        del args[CONCEPT_DATA_KEY]

        super().__init__(**args)

        # self.freeze()

        # self.verbose = phylo_args.get('verbose', False)

        # print model
        # print('totalmodel', self)
        # summary(self.cuda(), (1, 3, 512, 512))
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # if batch_idx%30==0 and batch_idx!=0:
        if batch_idx%30==0:
            print('cw module')
            self.eval()
            with torch.no_grad():                    
                for _, concept_batch in enumerate(self.concepts.train_dataloader()):
                    for idx, concept in enumerate(concept_batch['class']):
                        # print(concept.item())
                        X_var = torch.unsqueeze(concept_batch['image'][idx], dim=0)
                        concept_index = concept.item()
                        self.encoder.norm_out.mode = concept_index
                        X_var = torch.autograd.Variable(X_var).cuda()
                        X_var = X_var.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                        X_var = X_var.float()
                        self(X_var)
                        break
                # model.module.update_rotation_matrix()
                self.encoder.norm_out.update_rotation_matrix()
                # change to ordinary mode
                # model.module.change_mode(-1)
                self.encoder.norm_out.mode = -1
            self.train()


        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0 or (not self.loss.has_discriminator):
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1 and self.loss.has_discriminator:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss


class CWmodelVQGAN_baseline(VQModel):
    def __init__(self, **args):
        print(args)
        
        self.save_hyperparameters()

        # phylo_args = args[PHYLOCONFIG_KEY]
        # del args[PHYLOCONFIG_KEY]

        concept_data_args = args[CONCEPT_DATA_KEY]
        print("Concepts params : ", concept_data_args)
        self.concepts = instantiate_from_config(concept_data_args)
        self.concepts.prepare_data()
        self.concepts.setup()
        del args[CONCEPT_DATA_KEY]

        super().__init__(**args)
    
    def training_step(self, batch, batch_idx, optimizer_idx):

        n_cpt = 38 # number of concepts
        inter_feature = []
        def hookf(module, input, output):
            inter_feature.append(output[:,:n_cpt,:,:])
        if batch_idx%20==0:
            self.encoder.norm_out.register_forward_hook(hookf)

            y = []
            inter_feature = []
           
            for _, concept_batch in enumerate(self.concepts.train_dataloader()):
                for idx, concept in enumerate(concept_batch['class']):
                    # print(concept.item())
                    X_var = torch.unsqueeze(concept_batch['image'][idx], dim=0)
                    concept_index = concept.item()
                    self.encoder.norm_out.mode = concept_index
                    X_var = X_var.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)

                    y += [concept_index] * X_var.size(0)
                    X_var = torch.autograd.Variable(X_var).cuda()
                    X_var = X_var.float()
                    self(X_var)
                    break



        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0 or (not self.loss.has_discriminator):
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1 and self.loss.has_discriminator:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss