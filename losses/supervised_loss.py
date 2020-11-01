import numpy as np
import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path)
sys.path.append(os.path.dirname(curr_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_loss import BaseLoss

class SupervisedLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super(SupervisedLoss, self).__init__(*args, **kwargs)

    def forward(self, y_hat, y):
        pass

if __name__ == "__main__":
    loss_fn = SupervisedLoss("fake_config")
    print("done")