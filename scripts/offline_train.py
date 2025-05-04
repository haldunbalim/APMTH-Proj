

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks
from apmth import *
import torch.nn.functional as F
from hydra.utils import instantiate
import gymnasium as gym
import wandb
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="../config/offline", config_name="config")
def main(cfg: DictConfig):

    datamodule = TransitionDataModule(
        data_path=cfg.data_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    datamodule.setup()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    env = gym.make(cfg.env_name + '_train-v0')
    model = instantiate(cfg.model, env=env.unwrapped)

    wandb.init(project=cfg.wandb_project,
               sync_tensorboard=True, dir='.', config=cfg_dict)
    wandb.config.update(
        {'log_dir': HydraConfig.get().run.dir, 'script_name': os.path.basename(__file__)})
    tb_logger = pl_loggers.TensorBoardLogger('.')
    loggers = [tb_logger]

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}",
        every_n_epochs=cfg.every_n_epochs,
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator='auto', devices='auto',
                         logger=loggers, gradient_clip_algorithm='norm',
                         gradient_clip_val=1.0,
                         log_every_n_steps=300, max_epochs=cfg.n_epochs,
                         detect_anomaly=False, enable_checkpointing=cfg.every_n_epochs != 0)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()