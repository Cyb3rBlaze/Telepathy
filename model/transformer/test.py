import torch
from torch import nn

from utils import Transformer
from config import Config


# tests
random_input = (torch.rand(4, 5)*5).int()

config = Config()

model = Transformer(config)

print(model(random_input).shape)
