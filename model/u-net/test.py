import torch
from torch.nn import functional as F
import torch.optim as optim

from utils import UNet
from config import Config


# U-Net test function
def test_unet():
    config = Config()

    model = UNet(config)
    
    test_data = torch.rand((4, config.in_channels, 100, 100))
    print(test_data.shape)

    output = model(test_data)
    print(output.shape)

if __name__ == '__main__':
    test_unet()