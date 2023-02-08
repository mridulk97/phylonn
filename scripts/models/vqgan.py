#based on https://github.com/CompVis/taming-transformers


import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from scripts.import_utils import instantiate_from_config
from scripts.plotting_utils import dump_to_json
from scripts.modules.diffusionmodules.model import Encoder, Decoder
from scripts.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from scripts.modules.vqvae.quantize import GumbelQuantize
from scripts.modules.vqvae.quantize import EMAVectorQuantizer
from scripts.models.iterative_normalization import IterNormRotation as cw_layer

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 cw_module_transformers=False,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        
        # For wandb
        self.save_hyperparameters()
        
        self.cw_module_transformers = cw_module_transformers
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        if self.cw_module_transformers:
            self.encoder.norm_out = cw_layer(self.encoder.block_in)
            print("Changed to cw layer before loading cw model")
        
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.get_codebook_entry(code_b, shape=None)
        quant_b = quant_b.permute(1,0).view((1,quant_b.shape[0],16,16)) #TODO: this only works for vanilla vqgan feature map that is 16*16. Should eb generalized
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0 or (not self.loss.has_discriminator):
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            log_dict_ae["train/aeloss"] = aeloss
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            
            return aeloss

        if optimizer_idx == 1 and self.loss.has_discriminator:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            
            log_dict_disc["train/discloss"] = discloss
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        if self.loss.has_discriminator:
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                    last_layer=self.get_last_layer(), split="val")
            log_dict_disc["val/aeloss"] = aeloss
            log_dict_disc["val/rec_loss"] = log_dict_ae["val/rec_loss"]
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        else:
            log_dict_ae["val/aeloss"] = aeloss
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return self.log_dict

    @torch.no_grad()
    def on_validation_start(self):
        for v in self.trainer.val_dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)
    
    # NOTE: This is kinda hacky. But ok for now for test purposes.
    def set_test_chkpt_path(self, chkpt_path):
        self.test_chkpt_path = chkpt_path
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        rec_x, _ = self(x)
        
        return {
            'rec_loss': torch.abs(x.contiguous() - rec_x.contiguous())
        }
    
    @torch.no_grad()
    def test_epoch_end(self, in_out):
        rec_loss =torch.cat([x['rec_loss'] for x in in_out], 0)
        test_measures = {
            'rec_loss': torch.mean(rec_loss).item()
        }

        dump_to_json(test_measures, self.test_chkpt_path)

        return test_measures
    #######################################

    
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        r = self(x)
        xrec = r[0]
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
