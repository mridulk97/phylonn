from taming.loading_utils import load_config, load_phylovqvae, load_CWVQGAN
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate

from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
import torch


from omegaconf import OmegaConf
import argparse

###################

@torch.no_grad()
def main(configs_yaml):
    yaml_path= configs_yaml.yaml_path
    ckpt_path= configs_yaml.ckpt_path
    DEVICE= configs_yaml.DEVICE
    size= configs_yaml.size
    file_list_path= configs_yaml.file_list_path
    batch_size= configs_yaml.batch_size
    num_workers= configs_yaml.num_workers

    dataset = CustomDataset(size, file_list_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_CWVQGAN(config, ckpt_path=ckpt_path, data=dataset.data, cuda=(DEVICE is not None))
    # model.set_test_chkpt_path(ckpt_path)


    trainer = Trainer(distributed_backend='ddp', gpus='0,')
    test_measures = trainer.test(model, dataloader)

    print('test_measures', test_measures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/model_performance.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)