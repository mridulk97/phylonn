from scripts.constants import BASERECLOSS
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scripts.models.M_ModelAE_Cnn import CnnVae as LSFDisentangler

from main import instantiate_from_config

from scripts.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from scripts.models.vqgan import VQModel

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
        if 'ckpt_path' in LSF_args:
            del LSF_args['ckpt_path']

        self.LSF_disentangler = LSFDisentangler(**LSF_args)
        LSF_args['ckpt_path'] = ckpt_path

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
            print('Loaded trained model at', ckpt_path)

        self.verbose = LSF_args.get('verbose', False)

    def encode(self, x):
        encoder_out = self.encoder(x)
        disentangler_outputs = self.LSF_disentangler(encoder_out)
        disentangler_out = disentangler_outputs[DISENTANGLER_DECODER_OUTPUT]
        h = self.quant_conv(disentangler_out)
        quant, base_quantizer_loss, info = self.quantize(h)

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
    
    def forward_hypothetical(self, input):
        encoder_out = self.encoder(input)
        h = self.quant_conv(encoder_out)
        quant, base_hypothetical_quantizer_loss, info = self.quantize(h)
        dec = self.decode(quant)
        return dec, base_hypothetical_quantizer_loss

    def step(self, batch, batch_idx, prefix):
        x = self.get_input(batch, self.image_key)
        xrec, base_loss_dic, in_out_disentangler = self(x)

        if self.verbose:
            xrec_hypthetical, base_hypothetical_quantizer_loss = self.forward_hypothetical(x)
            hypothetical_rec_loss =torch.mean(torch.abs(x.contiguous() - xrec_hypthetical.contiguous()))
            self.log(prefix+"/base_hypothetical_rec_loss", hypothetical_rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            self.log(prefix+"/base_hypothetical_quantizer_loss", base_hypothetical_quantizer_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        # base losses
        true_rec_loss = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        self.log(prefix+  BASERECLOSS, true_rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log(prefix+"/base_quantizer_loss", base_loss_dic['quantizer_loss'], prog_bar=False, logger=True, on_step=False, on_epoch=True)

        total_loss, LSF_losses_dict = self.LSF_disentangler.loss(in_out_disentangler[DISENTANGLER_DECODER_OUTPUT], in_out_disentangler[DISENTANGLER_ENCODER_INPUT], 
                                                                     batch['class'], in_out_disentangler['embedding'], in_out_disentangler['vae_mu'], in_out_disentangler['vae_logvar'])
        
        if self.verbose:
            self.log(prefix+"/disentangler_total_loss", total_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            for i in LSF_losses_dict:
                if "_f1" in i:
                    self.log(prefix+"/disentangler_LSF_"+i, LSF_losses_dict[i], prog_bar=False, logger=True, on_step=True, on_epoch=True)

        self.log(prefix+"/disentangler_total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        for i in LSF_losses_dict:
            self.log(prefix+"/disentangler_LSF_"+i, LSF_losses_dict[i], prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log(prefix+"/disentangler_learning_rate", self.LSF_disentangler.learning_rate, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # monitor for checkpoint saving is set on this
        self.log(prefix+"/rec_loss", LSF_losses_dict['L_rec'], prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return total_loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')


    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
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
