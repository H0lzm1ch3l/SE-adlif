import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import logging
from models.pl_module_compress import MLPSNN

from pytorch_lightning.strategies import SingleDeviceStrategy
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors = matplotlib.colormaps.get_cmap('tab20').colors + matplotlib.colormaps.get_cmap('Set1').colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):        
    logging.getLogger().addHandler(logging.FileHandler("out.log"))
    logging.info(f"Experiment name: {cfg.exp_name}")
    pl.seed_everything(cfg.random_seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    
    model = MLPSNN(cfg)
    callbacks = []
    model_ckpt_tracker: ModelCheckpoint = ModelCheckpoint(
        monitor=cfg.get('tracking_metric', "val_acc_epoch"),
        mode=cfg.get('tracking_mode', 'max'),
        save_last=True,
        save_top_k=1,
        dirpath="ckpt"
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )
    callbacks = [model_ckpt_tracker, lr_monitor]
    logger = pl.loggers.CSVLogger("logs", name="mlp_snn")
    trainer: pl.Trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=cfg.n_epochs,
        enable_progress_bar=True,
        strategy=SingleDeviceStrategy(device=cfg.device),
        num_sanity_val_steps=1,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        )
    trainer.fit(model, datamodule=datamodule,)
    result = trainer.test(model, ckpt_path="best", datamodule=datamodule)
    logging.info(f"Final result: {result}")

    return trainer.checkpoint_callback.best_model_score.cpu().detach().numpy()


if __name__ == "__main__":
    main()
