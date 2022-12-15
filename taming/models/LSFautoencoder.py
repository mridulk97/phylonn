import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from taming.models.M_ModelAE_Cnn import CnnVae as LSFDisentangler

from main import instantiate_from_config

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.models.vqgan import VQModel

# from torchsummary import summary
from torchinfo import summary
import collections

import torchvision.utils as vutils


LSFLOCONFIG_KEY = "LSF_params"
BASEMODEL_KEY = "basemodel"

VQGAN_MODEL_INPUT = 'image'

DISENTANGLER_DECODER_OUTPUT = 'output'
DISENTANGLER_ENCODER_INPUT = 'in'
DISENTANGLER_CLASS_OUTPUT = 'class'
DISENTANGLER_ATTRIBUTE_OUTPUT = 'attribute'
DISENTANGLER_EMBEDDING = 'embedding'


class LSFVQVAE(VQModel):
    def __init__(self, **args):
        print(args)
        
        self.save_hyperparameters()

        LSF_args = args[LSFLOCONFIG_KEY]
        del args[LSFLOCONFIG_KEY]

        super().__init__(**args)

        self.freeze()

        ckpt_path = LSF_args.get('ckpt_path', None)
        if ckpt_path is not None:
            del LSF_args['ckpt_path']

        # self.LSF_disentangler = LSFDisentangler(**LSF_args)
        self.LSF_disentangler = LSFDisentangler(**LSF_args)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
            print('Loaded trained model at', ckpt_path)

        self.verbose = LSF_args.get('verbose', False)

    def encode(self, x):
        encoder_out = self.encoder(x)
        #phylo_quantizer_loss, classification_phylo_loss
        disentangler_outputs = self.LSF_disentangler(encoder_out)
        disentangler_out = disentangler_outputs[DISENTANGLER_DECODER_OUTPUT]
        h = self.quant_conv(disentangler_out)
        quant, base_quantizer_loss, info = self.quantize(h)

        # consolidate dicts
        base_loss_dic = {'quantizer_loss': base_quantizer_loss}
        in_out_disentangler = {
            DISENTANGLER_ENCODER_INPUT: encoder_out,
        }
        in_out_disentangler = {**in_out_disentangler, **disentangler_outputs}

        return quant, base_loss_dic, in_out_disentangler, info

    def forward(self, input):
        quant, base_loss_dic, in_out_disentangler, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, base_loss_dic, in_out_disentangler
    
    #TODO: make this only for debugging.
    def forward_hypothetical(self, input):
        encoder_out = self.encoder(input)
        h = self.quant_conv(encoder_out)
        quant, base_hypothetical_quantizer_loss, info = self.quantize(h)
        dec = self.decode(quant)
        return dec, base_hypothetical_quantizer_loss

    def step(self, batch, batch_idx, prefix):
        x = self.get_input(batch, self.image_key)
        # print('-'*10, 'x shape', x.shape)
        xrec, base_loss_dic, in_out_disentangler = self(x)
        out_disentangler = {i:in_out_disentangler[i] for i in in_out_disentangler if i not in [DISENTANGLER_ENCODER_INPUT, DISENTANGLER_DECODER_OUTPUT]}

        if self.verbose:
            xrec_hypthetical, base_hypothetical_quantizer_loss = self.forward_hypothetical(x)
            hypothetical_rec_loss =torch.mean(torch.abs(x.contiguous() - xrec_hypthetical.contiguous()))
            self.log(prefix+"/base_hypothetical_rec_loss", hypothetical_rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            self.log(prefix+"/base_hypothetical_quantizer_loss", base_hypothetical_quantizer_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        # base losses
        true_rec_loss = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        self.log(prefix+"/base_true_rec_loss", true_rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log(prefix+"/base_quantizer_loss", base_loss_dic['quantizer_loss'], prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # # # autoencode
        # # quantizer_disentangler_loss = disentangler_loss_dic['quantizer_loss']
        # # Calculates GAN discriminator loss - passed disentangler quantizer loss as 0
        # total_loss, log_dict_ae = self.LSF_disentangler.loss(torch.tensor([0], dtype=torch.float32), in_out_disentangler[DISENTANGLER_ENCODER_INPUT], in_out_disentangler[DISENTANGLER_DECODER_OUTPUT], 0, self.global_step, split=prefix)

        total_loss, LSF_losses_dict = self.LSF_disentangler.loss(in_out_disentangler[DISENTANGLER_DECODER_OUTPUT], in_out_disentangler[DISENTANGLER_ENCODER_INPUT], 
                                                                     batch['class'], in_out_disentangler['embedding'], in_out_disentangler['vae_mu'], in_out_disentangler['vae_logvar'])
        
        # if self.LSF_disentangler.loss_LSF is not None:
        #     LSF_losses_dict = self.LSF_disentangler.loss_LSF(total_loss, in_out_disentangler, self.LSF_disentangler.M, batch[DISENTANGLER_CLASS_OUTPUT], batch[VQGAN_MODEL_INPUT])
        #     total_loss = LSF_losses_dict['cumulative_loss']
        
        if self.verbose:
            self.log(prefix+"/disentangler_total_loss", total_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            for i in LSF_losses_dict:
                if "_f1" in i:
                    self.log(prefix+"/disentangler_LSF_"+i, LSF_losses_dict[i], prog_bar=False, logger=True, on_step=True, on_epoch=True)

        self.log(prefix+"/disentangler_total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        for i in LSF_losses_dict:
            self.log(prefix+"/disentangler_LSF_"+i, LSF_losses_dict[i], prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log(prefix+"/disentangler_learning_rate", self.LSF_disentangler.learning_rate, prog_bar=False, logger=True, on_step=False, on_epoch=True)


        # rec_loss = log_dict_ae[prefix+"/rec_loss"]
        # self.log(prefix+"/disentangler_quantizer_loss", quantizer_disentangler_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log(prefix+"/disentangler_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # if self.LSF_disentangler.loss_LSF is not None:
        #     self.log(prefix+"/disentangler_LSF_loss", LSF_losses_dict['total_LSF_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log(prefix+"/disentangler_LSF_L1_loss", LSF_losses_dict['LSF_L1_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log(prefix+"/disentangler_LSF_L2_loss", LSF_losses_dict['LSF_L2_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return total_loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')


    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
        # lr = self.learning_rate

        # opt_ae = torch.optim.Adam(list(self.phylo_disentangler.encoder.parameters())+
        #                         list(self.phylo_disentangler.decoder.parameters())+
        #                         list(self.phylo_disentangler.quantize.parameters())+
        #                         list(self.phylo_disentangler.quant_conv.parameters())+
        #                         list(self.phylo_disentangler.post_quant_conv.parameters()),
        #                         lr=lr, betas=(0.5, 0.9))
        # opt_ae = torch.optim.Adam(self.LSF_disentangler.parameters(), lr=lr, betas=(0.5, 0.9))

        # same as LSF originial implementation
        lr = self.LSF_disentangler.learning_rate
        opt_ae = torch.optim.Adam(self.LSF_disentangler.parameters(), lr=lr)
        
        return [opt_ae], []

    def image2encoding(self, x):
        encoder_out = self.encoder(x)
        mu, logvar = self.LSF_disentangler.encode(encoder_out)
        z = self.LSF_disentangler.reparameterize(mu, logvar)
        return z, mu, logvar

    def encoding2image(self, z):
        disentangler_out = self.LSF_disentangler.decoder(z)
        h = self.quant_conv(disentangler_out)
        quant, base_quantizer_loss, info = self.quantize(h)
        rec = self.decode(quant)
        return rec

    def image_translate(self, x, current_label, target_label, n_labels):
        # Based on the approach mentioned in paper
        quant, base_loss_dic, in_out_disentangler, _ = self.encode(x)
        z = in_out_disentangler[DISENTANGLER_EMBEDDING]
        U = self.LSF_disentangler.get_U()
        y_cap = z @ U.t()[:,:n_labels]
        s_cap = z @ U.t()[:,n_labels:]

        y_cap_current_pure = torch.zeros_like(y_cap)
        y_cap_current_pure[:,current_label] = 2
        z_current_pure = torch.cat([y_cap_current_pure, s_cap], -1) @ U

        y_cap_target = torch.zeros_like(y_cap)
        y_cap_target[:,target_label] = 2
        z_target = torch.cat([y_cap_target, s_cap], -1) @ U

        rec_current = self.encoding2image(z)
        rec_current_pure = self.encoding2image(z_current_pure)
        rec_target = self.encoding2image(z_target)#.add_(1.0).div_(2.0)

        return rec_current, rec_current_pure, rec_target


    def image_translate2(self, x, current_label, target_label, n_labels, target_percentage=1):
        quant, base_loss_dic, in_out_disentangler, _ = self.encode(x)
        z = in_out_disentangler[DISENTANGLER_EMBEDDING]

        U = self.LSF_disentangler.get_U()
        y_cap = z @ U.t()[:,:n_labels]
        s_cap = z @ U.t()[:,n_labels:]

        a = 0
        b = 2.6
        y_cap_new = torch.zeros_like(y_cap)
        y_cap_new[:,current_label] = a + ((b-a) * (1-target_percentage))
        y_cap_new[:,target_label] = a + ((b-a) * (target_percentage))
        z_new = torch.cat([y_cap_new, s_cap], -1) @ U

        rec_new = self.encoding2image(z_new)#.add_(1.0).div_(2.0)

        return rec_new


    # def predict(self, x, new_ls=None, weight=1.0):
    #     z, _ = self.encode(x)
    #     if new_ls is not None:
    #         zl = z @ self.M.t()
    #         d = torch.zeros_like(zl)
    #         for i, v in new_ls:
    #             d[:,i] = v*weight - zl[:,i]
    #         z += d @ self.M
    #     prod = self.decoder(z)
    #     return prod

    def predict(self, x, new_ls=None, weight=1.0):
        # Based on approach from LSF orig implementation
        quant, base_loss_dic, in_out_disentangler, _ = self.encode(x)
        disentangler_in = in_out_disentangler[DISENTANGLER_ENCODER_INPUT]
        disentangler_out = self.LSF_disentangler.predict(disentangler_in, new_ls)
        h = self.quant_conv(disentangler_out)
        quant, base_quantizer_loss, info = self.quantize(h)
        rec = self.decode(quant)
        return rec


    def generate(self, output_dir, labels, dataset, n_images=16, n_labels=38):
        label_dataset = collections.defaultdict(list)
        for d in dataset:
            if d['class'].item() in labels:
                label_dataset[d['class'].item()].append(d)

        U = self.LSF_disentangler.get_U()

        for label in label_dataset:
            d = label_dataset[label][0]
            x = d['image'].permute(0, 3, 1, 2)
            z, mu, logvar = self.image2encoding(x)
            s_cap = mu @ U.t()[:,n_labels:]
            y_cap = torch.zeros_like(z[:,:n_labels])
            y_cap[:,label] = 2
            z_new = torch.cat([y_cap, s_cap], -1) @ U
            imgs = []
            for i in range(n_images):
                z_gen = self.LSF_disentangler.reparameterize(z_new, logvar)
                gen_img = self.encoding2image(z_gen)
                imgs.append(gen_img.add(1.0).div(2.0).squeeze())

            img = torch.stack(imgs)
            vutils.save_image(img, f'{output_dir}/gen_with_{label}.jpg', nrow=4, padding=4)
            
            rec = self.encoding2image(z).add(1.0).div(2.0).squeeze()
            img = torch.stack([d['image'].permute(0, 3, 1, 2).add(1.0).div(2.0).squeeze(), rec])
            vutils.save_image(img, f'{output_dir}/orig-rec_with_{label}.jpg', nrow=4, padding=4)
