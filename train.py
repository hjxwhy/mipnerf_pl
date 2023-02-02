from fileinput import filename
import pytorch_lightning
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
#parser.add_argument("--data_path", help="data path.", type=str, required=True)
#parser.add_argument("--out_dir", help="Output directory.", type=str, required=True)
#parser.add_argument("--dataset_name", help="Single or multi data.", type=str, choices=['multi_blender', 'blender'],
#                    required=True)
#parser.add_argument("--config", help="Path to config file.", required=False, default='./configs/lego.yaml')
parser.add_argument("--config", help="Path to config file.", required=False, default='./configs/scannet.yaml')
parser.add_argument("opts", nargs=argparse.REMAINDER,
                    help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

class TorchTensorboardProfilerCallback(pytorch_lightning.Callback):
    """Quick-and-dirty Callback for invoking TensorboardProfiler during training.
    
    For greater robustness, extend the pl.profiler.profilers.BaseProfiler. See
    https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html"""

    def __init__(self, profiler):
        super().__init__()
        self.profiler = profiler 

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.profiler.step()
        #pl_module.log_dict(outputs)  # also logging the loss, while we're here

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

def main(hparams):
    import warnings
    from pytorch_lightning.profiler import PyTorchProfiler, AdvancedProfiler, SimpleProfiler
    from pytorch_lightning.profiler import Profiler
    from torch.profiler import tensorboard_trace_handler
    import torch.profiler
    from pytorch_lightning.callbacks import DeviceStatsMonitor
    warnings.filterwarnings('ignore')
    torch.autograd.set_detect_anomaly(False)



    setup_seed(hparams['seed'])
    system = MipNeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(hparams['out_dir'], 'ckpt', hparams['exp_name']),
                              save_last=True,
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=2,
                              filename='epoch-{epoch:02d}_step-{step}_psnr-val_psnr{val/psnr:.4f}',
                              auto_insert_metric_name=False
                              )
    pbar = TQDMProgressBar(refresh_rate=1)
    

    logger = TensorBoardLogger(save_dir=os.path.join(hparams['out_dir'], "logs"),
                               name=hparams['exp_name'],
                               default_hp_metric=False)
    
    #profiler = PyTorchProfiler(filename="performance.txt", record_functions={'training_step', 'backward', 'optimizer_step'},
    #    with_stack=True,
    #    export_to_chrome=True)
        #emit_nvtx=True)
        
    #profiler = SimpleProfiler(filename="performance.txt", extended=True)
    #profiler = torch.profiler.profiler(with_stack=True, with_modules=True, profile_memory=True)
    
    #profiler = AdvancedProfiler(filename="performance.txt")
    #profiler = torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler("name=hparams['exp_name']"),
    #   profile_memory=True, with_stack=True)

    callbacks = [ckpt_cb] #,pbar, DeviceStatsMonitor(),

    #profiler = torch.profiler.profile(with_stack=True, profile_memory=True, with_modules=True,
    #        schedule=torch.profiler.schedule(wait=1,warmup=400, active=3, repeat=1, skip_first=0),
    #        on_trace_ready=tensorboard_trace_handler(f"{hparams['out_dir']}/logs/{hparams['exp_name']}"))
    #with profiler:
    #    profiler_callback = TorchTensorboardProfilerCallback(profiler)
    #    callbacks = [ckpt_cb, profiler_callback] #,pbar, DeviceStatsMonitor(), 
    #        
    trainer = Trainer(
        max_steps=hparams['optimizer.max_steps'],
        max_epochs=-1,
        callbacks=callbacks,
        val_check_interval=hparams['val.check_interval'] * hparams['accumulate_grad_batches'],
        logger=logger,
        enable_model_summary=False,
        accelerator='gpu',
        devices=hparams['num_gpus'],
        num_sanity_val_steps=1,
        benchmark=True,
        #profiler='pytorch' if hparams['num_gpus'] == 1 else None,
        precision=hparams['precision'],
        #strategy=DDPPlugin(find_unused_parameters=False) if hparams['num_gpus'] > 1 else None,
        accumulate_grad_batches=hparams['accumulate_grad_batches'],
        check_val_every_n_epoch=None,
        limit_val_batches=hparams['val.sample_num'],
        enable_progress_bar=False,
        log_every_n_steps=200,
        gradient_clip_val=1e-3,
    )

    trainer.fit(system, ckpt_path=hparams['checkpoint.resume_path'])
    #   profiler.export_chrome_trace("trace.json")


if __name__ == "__main__":
    main(parse_args(parser))
