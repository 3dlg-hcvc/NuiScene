import os
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

from src.model.vae.model import VAEModel
from src.data.data_module import DataModuleNoVal

def init_callbacks(cfg):
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    return [checkpoint_monitor, lr_monitor]

@hydra.main(version_base=None, config_path="configs", config_name="diff_config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # load the vae model
    vae = VAEModel.load_from_checkpoint(cfg.vae_ckpt_path)

    # create directories for training outputs
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    # initialize data
    data_module = DataModuleNoVal(cfg.data)

    # initialize model
    model = hydra.utils.instantiate(cfg.model_diff.model_name, cfg, vae=vae)

    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger)

    # initialize callbacks
    callbacks = init_callbacks(cfg)

    # initialize trainer
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."

    # start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)

if __name__ == '__main__':
    main()
