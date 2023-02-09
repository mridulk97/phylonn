import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import pytorch_lightning as p
from main import instantiate_from_config

from taming.models.vqgan import VQModel

from taming.models.iterative_normalization import IterNormRotation as cw_layer


from taming.analysis_utils import get_CosineDistance_matrix, aggregate_metric_from_specimen_to_species
from taming.plotting_utils import plot_heatmap

# from torchsummary import summary
from torchinfo import summary
import collections
import itertools

CONCEPT_DATA_KEY = "concept_data"
activation_mode = 'pool_max'


class CWmodelVQGAN(VQModel):
    def __init__(self, **args):
        print(args)
        
        self.save_hyperparameters()

        concept_data_args = args[CONCEPT_DATA_KEY]
        print("Concepts params : ", concept_data_args)
        self.concepts = instantiate_from_config(concept_data_args)
        self.concepts.prepare_data()
        self.concepts.setup()
        del args[CONCEPT_DATA_KEY]


        super().__init__(**args)
        
        if not self.cw_module_transformers:
            self.encoder.norm_out = cw_layer(self.encoder.block_in)
            print("Changed to cw layer after loading base VQGAN")

        # self.freeze()

        # self.verbose = phylo_args.get('verbose', False)

        # print model
        # print('totalmodel', self)
        # summary(self.cuda(), (1, 3, 512, 512))
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        if (batch_idx+1)%30==0 and optimizer_idx==0:
            print('cw module')
            self.eval()
            with torch.no_grad():                    
                for _, concept_batch in enumerate(self.concepts.train_dataloader()):
                    for idx, concept in enumerate(concept_batch['class'].unique()):
                        concept_index = concept.item()
                        self.encoder.norm_out.mode = concept_index
                        X_var = concept_batch['image'][concept_batch['class'] == concept]
                        X_var = X_var.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                        X_var = torch.autograd.Variable(X_var).cuda()
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
            

    @torch.no_grad() #TODO: maybe put this for all analysis scritps?
    def test_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        h = self.encoder(x)
        h = self.quant_conv(h)
        class_label = batch['class']

        return {'z_cw': h,
                'label': class_label,
                'class_name': batch['class_name']}

    # NOTE: This is kinda hacky. But ok for now for test purposes.
    def set_test_chkpt_path(self, chkpt_path):
        self.test_chkpt_path = chkpt_path

    @torch.no_grad()
    def test_epoch_end(self, in_out):
        z_cw =torch.cat([x['z_cw'] for x in in_out], 0)
        labels =torch.cat([x['label'] for x in in_out], 0)
        sorting_indices = np.argsort(labels.cpu())
        sorted_zq_cw = z_cw[sorting_indices, :]
        
        z_cosine_distances = get_CosineDistance_matrix(sorted_zq_cw)
        plot_heatmap(z_cosine_distances.cpu(), '/home/mridul/taming-transformers/paper_plots/', title='Cosine_distances_tinker', postfix='test_data_infer_model')

        classnames = list(itertools.chain.from_iterable([x['class_name'] for x in in_out]))
        sorted_class_names_according_to_class_indx = [classnames[i] for i in sorting_indices]
        z_cosine_distances_avg_over_classes = aggregate_metric_from_specimen_to_species(sorted_class_names_according_to_class_indx, z_cosine_distances)
       
        # phylo_dist = aggregate_metric_from_specimen_to_species(sorting_indices, z_cosine_distances)
        plot_heatmap(z_cosine_distances_avg_over_classes.cpu(), '/home/mridul/taming-transformers/paper_plots', title='Cosine_distances_aggregated_tinker', postfix='test_data_infer_model')

        z_cosine_distances_avg_over_classes_np = z_cosine_distances_avg_over_classes.cpu().numpy()
        df_avg = pd.DataFrame(z_cosine_distances_avg_over_classes_np)
        df_avg.to_csv("/home/mridul/figs/test_data_infer_model/CW_z_cosine_distances_avg_over_classes.csv",index=False)

        z_cosine_distancess_np = z_cosine_distances.cpu().numpy()
        df = pd.DataFrame(z_cosine_distancess_np)
        df.to_csv("/home/mridul/figs/test_data_infer_model/CW_z_cosine_distances.csv",index=False) 
        
        return None


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __init__(self, **args):
        print(args)
        
        self.save_hyperparameters()


        concept_data_args = args[CONCEPT_DATA_KEY]
        print("Concepts params : ", concept_data_args)
        self.concepts = instantiate_from_config(concept_data_args)
        self.concepts.prepare_data()
        self.concepts.setup()
        del args[CONCEPT_DATA_KEY]

        super().__init__(**args)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        loss_aux = AverageMeter()
        top1_cpt = AverageMeter()
        n_cpt = 38 # number of concepts
        inter_feature = []
        def hookf(module, input, output):
            inter_feature.append(output[:,:n_cpt,:,:])
        if (batch_idx+1)%20==0 and optimizer_idx == 0:
            hook = self.encoder.norm_out.register_forward_hook(hookf)

            y = []
            inter_feature = []
           
            for i, concept_batch in enumerate(self.concepts.train_dataloader()):    
                for idx, concept in enumerate(concept_batch['class'].unique()):
                    concept_index = concept.item()
                    self.encoder.norm_out.mode = concept_index
                    X_var = concept_batch['image'][concept_batch['class'] == concept]
                    X_var = X_var.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                    y += [concept_index] * X_var.size(0)
                    X_var = torch.autograd.Variable(X_var).cuda()
                    X_var = X_var.float()
                    self(X_var)
                    break
            
            inter_feature = torch.cat(inter_feature,0)
            y_var = torch.Tensor(y).long().cuda()
            f_size = inter_feature.size()
            if activation_mode == 'mean':
                y_pred = F.avg_pool2d(inter_feature,f_size[2:]).squeeze()
            elif activation_mode == 'max':
                y_pred = F.max_pool2d(inter_feature,f_size[2:]).squeeze()
            elif activation_mode == 'pos_mean':
                y_pred = F.avg_pool2d(F.relu(inter_feature),f_size[2:]).squeeze()
            elif activation_mode == 'pool_max':
                kernel_size = 3
                y_pred = F.max_pool2d(inter_feature, kernel_size)
                y_pred = F.avg_pool2d(y_pred,y_pred.size()[2:]).squeeze()
            
            criterion_cw = nn.CrossEntropyLoss().cuda()
            momentum_cw = 0.9
            learn_rate_cw = 0.05
            weight_decay_cw = 1e-4
            optimizer_SGD = torch.optim.SGD(self.parameters(), learn_rate_cw,
                        momentum=momentum_cw,
                        weight_decay=weight_decay_cw)
            loss_cpt = 10*criterion_cw(y_pred, y_var)
            # measure accuracy and record loss
            [prec1_cpt] = accuracy(y_pred.data, y_var, topk=(1,))
            loss_aux.update(loss_cpt.data, f_size[0])
            top1_cpt.update(prec1_cpt[0], f_size[0])
            
            optimizer_SGD.zero_grad()
            loss_cpt.backward()
            optimizer_SGD.step()

            hook.remove()
            loss_log_cw = {"{}/cw_loss_val".format('train'): loss_aux.val,
                            "{}/cw_loss_avg".format('train'): loss_aux.avg,
                            "{}/cw_top1_cpt_val".format('train'): top1_cpt.val,
                            "{}/cw_top1_cpt_avg".format('train'): top1_cpt.avg,
                            }
            # self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(loss_log_cw, prog_bar=True, logger=True, on_step=True, on_epoch=True)



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