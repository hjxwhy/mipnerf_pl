from models.nerf_system import MipNeRFSystem
# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import argparse
import os
from configs.config import parse_args
import torch
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", help="Output directory.", type=str, required=True)
parser.add_argument("--config", help="Path to config file.", required=False, default='./configs/lego.yaml')
parser.add_argument("opts", nargs=argparse.REMAINDER,
                    help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(hparams):
    setup_seed(hparams['seed'])
    system = MipNeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(hparams['out_dir'], 'ckpt', hparams['exp_name']),
                              # filename='{epoch:03d}',
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=2,
                              # every_n_train_steps=10000
                              )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir=os.path.join(hparams['out_dir'], "logs"),
                               name=hparams['exp_name'],
                               default_hp_metric=False)

    trainer = Trainer(
        gradient_clip_algorithm='norm',
        max_steps=hparams['optimizer.max_steps'],
        max_epochs=-1,
        callbacks=callbacks,
        val_check_interval=hparams['val.check_interval'],
        logger=logger,
        enable_model_summary=False,
        accelerator='auto',
        devices=hparams['num_gpus'],
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if hparams['num_gpus'] == 1 else None,
        strategy=DDPPlugin(find_unused_parameters=False) if hparams['num_gpus'] > 1 else None,
        limit_val_batches=hparams['val.sample_num']
    )

    trainer.fit(system, ckpt_path=hparams['checkpoint.resume_path'])


if __name__ == "__main__":
    main(parse_args(parser))
