
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import set_optimizer_mom
from transformers import get_linear_schedule_with_warmup
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler


def cosine_annealing(it, n_iter, start_val, end_val):
    cos_inner = math.pi * (it % n_iter) / n_iter
    return ((start_val - end_val) * (math.cos(cos_inner) + 1) / 2) + end_val


def cosine_annealing_range(n_iter, start_val, end_val):
    return [cosine_annealing(i, n_iter, start_val, end_val) 
            for i in range(n_iter)]


class OneCycleLR(LambdaLR):
    def __init__(self, optimizer, lr_div_factor=25, warmup_frac=0.3, 
                 mom_range=(0.95, 0.85), n_epochs=10, n_batches=None, 
                 start_epoch=0):
        n_batches = 1 if n_batches is None else n_batches
        self.n_epochs, self.n_iter = n_epochs, (n_epochs * n_batches) + 1
        self.start_it = -1 if start_epoch==0 else start_epoch * n_batches
        self._build_schedules(lr_div_factor, mom_range, warmup_frac)
        super().__init__(optimizer, self.lr_lambda, last_epoch=self.start_it)
        
    def _build_schedules(self, lr_div_factor, mom_range, warmup_frac):
        n_warmup = int(self.n_iter * warmup_frac)
        n_decay = self.n_iter - n_warmup
        
        self.lrs = cosine_annealing_range(n_warmup, 1/lr_div_factor, 1)
        self.lrs += cosine_annealing_range(n_decay, 1, 1/lr_div_factor)
        self.lr_lambda = lambda i: self.lrs[i]
        
        self.moms = cosine_annealing_range(n_warmup, *mom_range)
        self.moms += cosine_annealing_range(n_decay, *mom_range[::-1])
        self.mom_lambda = lambda i: self.moms[i]
        
    def get_mom(self):
        return self.mom_lambda(self.last_epoch)

    def step(self, epoch=None):
        super().step(epoch)
        set_optimizer_mom(self.optimizer, self.get_mom())
        
    def plot_schedules(self):
        x = np.linspace(0, self.n_epochs, self.n_iter)
        _, ax = plt.subplots(1, 2, figsize=(15, 4))

        ax[0].set_title('LR Schedule')
        ax[0].set_ylabel('lr')
        ax[0].set_xlabel('epoch')
        ax[0].plot(x, self.lrs)

        ax[1].set_title('Momentum Schedule')
        ax[1].set_ylabel('momentum')
        ax[1].set_xlabel('epoch')
        ax[1].plot(x, self.moms)


class CosineAnnealingWithRestartScheduler(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cyclic cosine annealing schedule
    :math:`T` is total number of steps in the training (:math:`total_epochs * steps_per_epoch`),
    :math:`M` is the number of cycles wanted in the training,
    :math:`lr_max` is the initial cycle lr and :math:`lr_min` the final cycle lr.
    
    At next cycles, new lr is computed with method compute_lr_min_max from from lr_min, lr_max, last_epoch and cur_cycle that you can override if you want dynamic lr_min/lr_max depending on epoch and cycle.
    .. math::
        \alpha(t) = {\lr_min} + \frac ({\lr_max} - {\lr_min}) {2} (cos (\pi * \frac {mod(t - 1, \lceil\frac {T} {M}\rceil} {\frac {T} {M}}) + 1)
    When last_epoch=-1, sets initial lr as lr_max.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_min (float): the minimum learning rate (at end of cycle).
        lr_max (float): the maximum learning rate (at start of cycle).
        total_epochs (int): The total number of epochs in training
        steps_per_epoch (int): The total number of steps per epoch (len(dataloader)).
        nb_cycles (int): The number of cycles in the cyclic annealing.
        last_epoch (int): The index of last step (it is not an epoch in . Default: -1.
        new_lr_min_max (lambda): a lambda computing new lr_min and lr_max from lr_min, lr_max, last_epoch and cur_cycle, by default using same lr_min and lr_max
    .. Inspired by
        - Snapshot Ensembles: Train 1, get M for free https://arxiv.org/abs/1704.00109
        - SGDR : Stochastic Gradient Descent with Warm Restarts https://arxiv.org/abs/1608.03983
        - Pytorch CosineAnnealing LR Scheduler https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """

    def __init__(self, optimizer,
                 lr_min, lr_max,
                 total_epochs, steps_per_epoch, nb_cycles,                
                 last_epoch=-1):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.total_epochs = total_epochs
        self.nb_cycles = nb_cycles
        self.steps_per_epoch = steps_per_epoch
        
        self.steps_per_cycle = math.ceil((self.total_epochs * self.steps_per_epoch) / self.nb_cycles)
        # current cycle
        self.cur_cycle = 1
        self.cur_lr_min, self.cur_lr_max = self.compute_lr_min_max(self.lr_min, self.lr_max, last_epoch, self.cur_cycle)
        self.cur_lr = self.cur_lr_max
        
        super(CosineAnnealingWithRestartScheduler, self).__init__(optimizer, last_epoch)
    
    def compute_lr_min_max(self, lr_min, lr_max, last_epoch, cur_cycle):
        return (lr_min, lr_max)
    
    def get_lr(self):
        # in snapshot ensemble, last_epoch is not about epochs but batch step as we update LR at each iteration
        cur_step = self.last_epoch
        
        # update cycle if reached steps_per_cycle steps in current cycle
        if cur_step > self.cur_cycle * self.steps_per_cycle:
            self.cur_cycle += 1
            self.cur_lr_min, self.cur_lr_max = self.compute_lr_min_max(self.lr_min, self.lr_max, self.last_epoch, self.cur_cycle)
            
        self.cur_lr = self.cur_lr_min + 0.5 * (self.cur_lr_max - self.cur_lr_min) * (1 + math.cos(math.pi * (cur_step % self.steps_per_cycle) / self.steps_per_cycle))
        
        # not using base_lr here
        return [
            self.cur_lr
            for base_lr in self.base_lrs
        ]

def snapshot_ensemble_scheduler(optimizer, lr_max, epochs, nb_cycles, train_loader):
    scheduler = CosineAnnealingWithRestartScheduler(optimizer,
                                                    lr_min=0., lr_max=lr_max,
                                                    total_epochs=epochs,
                                                    nb_cycles=nb_cycles,
                                                    steps_per_epoch=len(train_loader)
                                                    )
    return scheduler





def get_scheduler(args, optimizer, last_epoch, train_loader):
        assert args.scheduler_name in ['multistep', 'linear_warmup', 'onecycle', 'snapshot_ensemble_scheduler','cyclic', None]
        if args.scheduler_name is None:
            return None
        if args.scheduler_name == 'multistep':
            return lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20,30], gamma=0.1, last_epoch=last_epoch)
        
        if args.scheduler_name == 'linear_warmup':
            # Total number of training steps is [number of batches] x [number of epochs]. 
            total_steps = len(train_loader) * args.num_epochs
            warmup_frac = 0.3
            return get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = int(total_steps * warmup_frac), 
                                            num_training_steps = total_steps)
        if args.scheduler_name == 'onecycle':
            return OneCycleLR(optimizer, n_epochs=args.num_epochs, n_batches=len(train_loader))
        if args.scheduler_name == 'snapshot_ensemble_scheduler':
            nb_cycles = 2
            return snapshot_ensemble_scheduler(optimizer, args.lr, args.num_epochs, nb_cycles, train_loader)
        if args.scheduler_name == 'cyclic':
            return lr_scheduler.CyclicLR(optimizer, 0.05, 0.01)

  

  