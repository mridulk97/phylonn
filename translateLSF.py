import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only

import torchvision.utils as vutils
from tqdm import tqdm
from taming.data.utils import custom_collate

import wandb

# from analysis.test_resnet_translation import classification_all_levels
# import gc
import ntpath

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    # parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--classification_config",
        type=str,
        nargs="?",
        const=True,
        default=None,
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):

        #TODO: check when this is fixed by wandb

        # grids = dict()
        # for k in images:
        #     grid = torchvision.utils.make_grid(images[k])
        #     # print('grid', grid.shape, grid.dtype)
        #     grid = grid.permute(1, 2, 0)
        #     grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
        #     # print('grid', grid.shape, grid.dtype)
        #     grids[f"{split}/{k}"] = wandb.Image(grid.numpy())
        # # print(grids)
        # pl_module.logger.experiment.log(grids)
        # # pl_module.logger.experiment.log({
        # #     "samples": [wandb.Image(grids[key]) for key in grids]
        # #     })

        for k in images:
            grid = torchvision.utils.make_grid(images[k].detach())
            # print('grid', grid.shape, grid.dtype)
            grid = grid.cpu().numpy().transpose((1, 2, 0))
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = (grid * 255).astype(np.uint8)
            grid = Image.fromarray(grid)
            # print('grid', grid.shape, grid.dtype)
            # grids[f"{split}/{k}"] = wandb.Image(grid)
        # print(grids)
            pl_module.logger.experiment.log({f"{split}/{k}": wandb.Image(grid)})
            # pl_module.logger.experiment.log({"hi": wandb.Image(grid)})
        # pl_module.logger.experiment.log({
        #     "samples": [wandb.Image(grids[key]) for key in grids]
        #     })
    



    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join("logs", nowname)
    os.makedirs(logdir) #NOTE: needed to fix wandb image generation logging

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    cli2 = OmegaConf.from_cli()
    config = OmegaConf.merge(*configs, cli2)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["distributed_backend"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["distributed_backend"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)

    # from data.fish_labels_to_idx import labels_to_idx

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    val_data = data.val_dataloader()
    train_data = data.train_dataloader()

    current_label = 15
    target_label = 13
    n_labels = 38

    def postprocess(grid):
        grid = grid.cpu().detach().numpy().transpose((0, 2, 3, 1))
        grid = grid.squeeze(0)
        grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
        grid = (grid * 255).astype(np.uint8)
        return grid

    
    def label_morphing(output_dir, base_img_path, test_data, current_label, target_label, step=5, save=1, n_imgs=None):
        filtered_dataset = []
        for d in test_data:
            if (d['class'].item() == current_label) and (d['file_path_'][0] == base_img_path.split('.')[0]):
                filtered_dataset.append(d)

        if not os.path.isdir(f'{output_dir}/Morphing_Labels/'):
            os.makedirs(f'{output_dir}/Morphing_Labels/')
        
        total_it = len(filtered_dataset)
        bar_data = tqdm(filtered_dataset, total=total_it)
        all_imgs = []
        for i, d in enumerate(bar_data):
            imgs = []
            imgs.append(d['image'].permute(0, 3, 1, 2).add(1.0).div(2.0).squeeze())
            rec_orig, _, _ = model(d['image'].permute(0, 3, 1, 2))
            rec_orig = rec_orig.add(1.0).div(2.0).squeeze()
            imgs.append(rec_orig)
            with torch.no_grad():
                for s in range(step):
                    prod = model.image_translate2(d['image'].permute(0, 3, 1, 2), 
                                            current_label, target_label, n_labels=38, 
                                            target_percentage=(s+1)/step)
                    prod = prod.squeeze()
                    prod.add_(1.0).div_(2.0)
                    imgs.append(prod)
                if save == 1:
                    img = torch.stack(imgs)
                    vutils.save_image(img, f'{output_dir}/Morphing_Labels/progressive_morphing_{current_label}-to-{target_label}-{i}.jpg', nrow=step+2, padding=4)
            if save == 1:
                img = torch.stack([imgs[0], imgs[1], imgs[-1]])
                vutils.save_image(img, f'{output_dir}/Morphing_Labels/morphing_img_{current_label}-to-{target_label}-{i}.jpg',padding=0)

            all_imgs.append(imgs)
            if n_imgs is not None and (i+1) == n_imgs:
                break

        return all_imgs


    def pic_morphing(output_dir, base_img_path, target_img_path, test_data, step=5):

        base_img_path = os.path.splitext(base_img_path)[0]
        target_img_path = os.path.splitext(target_img_path)[0]

        for d in test_data:
            path = os.path.splitext(d['file_path_'][0])[0]
            if (path == target_img_path):
                target_img = d['image'].permute(0, 3, 1, 2)
            elif (path == base_img_path):
                base_img = d['image'].permute(0, 3, 1, 2)      

        if not os.path.isdir(f'{output_dir}/Morphing_Pics/'):
            os.makedirs(f'{output_dir}/Morphing_Pics/')
        imgs = []
        with torch.no_grad():
            data1 = base_img#.to("cuda:0")
            imgs.append(data1.add(1.0).div(2.0).squeeze())
            data2 = target_img#.to("cuda:0")
            z1,_, _ = model.image2encoding(data1)
            z2,_, _ = model.image2encoding(data2)
            for s in range(1, step-1):
                z = (z1*(step-1-s)+z2*s)/(step-1)
                prod = model.encoding2image(z)
                prod.add_(1.0).div_(2.0)
                imgs.append(prod.squeeze())
            vutils.save_image(data1.add(1.0).div(2.0), f'{output_dir}/Morphing_Pics/{ntpath.basename(base_img_path)}.jpg',padding=0)
            vutils.save_image(data2.add(1.0).div(2.0), f'{output_dir}/Morphing_Pics/{ntpath.basename(target_img_path)}.jpg',padding=0)
        imgs.append(data2.add(1.0).div(2.0).squeeze())
        img = torch.stack(imgs)
        vutils.save_image(img, f'{output_dir}/Morphing_Pics/Morphing_Pics.jpg', nrow=step, padding=4)

        return imgs

    output_dir = '/home/harishbabu/projects/taming-transformers/translations18'
    base_img_path = '/home/harishbabu/data/Fish/phylo-VQVAE/train_padded_256/Lepomis humilis/INHS_FISH_13404.jpg'
    target_img_path='/home/harishbabu/data/Fish/phylo-VQVAE/train_padded_256/Gambusia affinis/INHS_FISH_32786.JPG'
    # output_dir = 'translations19'
    # base_img_path = 'data/Fish/phylo-VQVAE/train_padded_256/Lepomis humilis/INHS_FISH_13404.jpg'
    # target_img_path='data/Fish/phylo-VQVAE/train_padded_256/Gambusia affinis/INHS_FISH_32786.JPG'
    imgs = pic_morphing(output_dir, base_img_path, target_img_path, test_data=train_data, step=7)