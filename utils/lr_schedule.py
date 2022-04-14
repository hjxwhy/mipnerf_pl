import torch
import numpy as np


class MipLRDecay(torch.optim.lr_scheduler._LRScheduler):
    """
    Continuous learning rate decay function.
        The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
        is log-linearly interpolated elsewhere (equivalent to exponential decay).
        If lr_delay_steps>0 then the learning rate will be scaled by some smooth
        function of lr_delay_mult, such that the initial learning rate is
        lr_init*lr_delay_mult at the beginning of optimization but will be eased back
        to the normal learning rate when steps>lr_delay_steps.
    Args:
        step: int, the current optimization step.
        lr_init: float, the initial learning rate.
        lr_final: float, the final learning rate.
        max_steps: int, the number of steps during optimization.
        lr_delay_steps: int, the number of steps to delay the full learning rate.
        lr_delay_mult: float, the multiplier on the rate when delaying it.
    Returns:
        lr: the learning for current step 'step'.
    """
    def __init__(self, optimizer,
                 lr_init: float,
                 lr_final: float,
                 max_steps: int,
                 lr_delay_steps: int,
                 lr_delay_mult: float):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        super(MipLRDecay, self).__init__(optimizer)


    # def step(self, optimizer, step):
    #     if self.lr_delay_steps > 0:
    #         # A kind of reverse cosine decay.
    #         delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
    #             0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
    #     else:
    #         delay_rate = 1.
    #     t = np.clip(step / self.max_steps, 0, 1)
    #     log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
    #     # return delay_rate * log_lerp
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = delay_rate * log_lerp

    def get_lr(self):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(self.last_epoch / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(self.last_epoch / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return [delay_rate * log_lerp]
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = delay_rate * log_lerp