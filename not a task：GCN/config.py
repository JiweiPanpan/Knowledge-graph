"""
Laboratory Artificial Intelligence: Deep Learning Lab
Created on Jun 10, 2023
@Team: 02 Jiwei Pan, Ziming Fang
"""

import torch
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MARGIN = 0.5
MARGIN_NEG = 1.5