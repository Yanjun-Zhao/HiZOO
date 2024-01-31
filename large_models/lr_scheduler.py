# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
#from torch.optim import Optimizer
#from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from transformers.utils import logging
from transformers.utils.versions import require_version


logger = logging.get_logger(__name__)


def get_constant_schedule(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):

    return learning_rate


def get_reduce_on_plateau_schedule(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):

    #to be updated
    return learning_rate

def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay


def get_inverse_sqrt_schedule(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):
    #to be updated
    return learning_rate

def get_constant_schedule_with_warmup(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):
    
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps)) * learning_rate
    
    return learning_rate


def get_linear_schedule_with_warmup(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):


    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * learning_rate
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))) * learning_rate




def get_cosine_schedule_with_warmup(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):

    num_cycles=2
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * learning_rate
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * learning_rate



def get_cosine_with_hard_restarts_schedule_with_warmup(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):
    num_cycles=2
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * learning_rate
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0 * learning_rate
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))) * learning_rate




def get_polynomial_decay_schedule_with_warmup(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):
    
    lr_init = learning_rate
    lr_end = 1e-10
    power = 3
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * learning_rate
    elif current_step > num_training_steps:
        return lr_end / lr_init * learning_rate  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init * learning_rate  # as LambdaLR multiplies by lr_init



def get_polynomial_decay_schedule(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):
    
    lr_init = learning_rate
    lr_end = 1e-10
    power = 3
    if current_step > num_training_steps:
        return lr_end / lr_init * learning_rate  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - 0
        pct_remaining = 1 - (current_step - 0) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init * learning_rate  # as LambdaLR multiplies by lr_init
        
def get_constant_polynomial_decay_schedule(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):
    lr_init = learning_rate
    lr_end = 1e-10
    power = 3
    if current_step < num_decay_steps:
        return learning_rate
    elif current_step > num_training_steps:
        return lr_end / lr_init * learning_rate  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_decay_steps
        pct_remaining = 1 - (current_step - num_decay_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init * learning_rate / 100  # as LambdaLR multiplies by lr_init

def get_constants_schedule(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):
    if current_step < num_decay_steps:
        return learning_rate
    else:
        return learning_rate/10
    


def get_constant_polynomial_decay_schedule_with_warmup(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):
    lr_init = learning_rate
    lr_end = 1e-7
    power = 3
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * learning_rate
    elif current_step >= num_warmup_steps and current_step < num_decay_steps:
        return learning_rate
    elif current_step > num_training_steps:
        return lr_end / lr_init * learning_rate  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_decay_steps
        pct_remaining = 1 - (current_step - num_decay_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init * learning_rate  # as LambdaLR multiplies by lr_init




TYPE_TO_SCHEDULER_FUNCTION = {
    'linear_with_warmup': get_linear_schedule_with_warmup,
    'cosine_with_warmup': get_cosine_schedule_with_warmup,
    'cosine_with_hard_restarts_with_warmup': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial_decay_with_warmup': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_schedule_with_warmup,
    'inverse_sqrt': get_inverse_sqrt_schedule,
    'reduce_on_plateau': get_reduce_on_plateau_schedule,
    'constant_polynomial_decay' : get_constant_polynomial_decay_schedule,
    'constant_polynomial_decay_with_warmup' : get_constant_polynomial_decay_schedule_with_warmup,
    'polynomial_decay' : get_polynomial_decay_schedule,
    'constants': get_constants_schedule,
    
}



def zo_lr_scheduler(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps):

    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    return schedule_func(learning_rate, name, num_warmup_steps, num_decay_steps, current_step, num_training_steps)
