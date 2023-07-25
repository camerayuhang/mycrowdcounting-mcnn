import torch
from torch import nn
from torch.utils.data import DataLoader


class mcnn(nn.Module):
  def __init__(self, in_channels) -> None:
    super().__init__()
    self.branch1 = nn.Sequential(
        nn.Conv2d(in_channels, 16, 9, padding="same"), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, 7, padding="same"), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 16, 7, padding="same"), nn.ReLU(),
        nn.Conv2d(16, 8, 7, padding="same"), nn.ReLU()
    )
    self.branch2 = nn.Sequential(
        nn.Conv2d(in_channels, 20, 7, padding="same"), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(20, 40, 5, padding="same"), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(40, 20, 5, padding="same"), nn.ReLU(),
        nn.Conv2d(20, 10, 5, padding="same"), nn.ReLU()
    )
    self.branch3 = nn.Sequential(
        nn.Conv2d(in_channels, 24, 5, padding="same"), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(24, 48, 3, padding="same"), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(48, 24, 3, padding="same"), nn.ReLU(),
        nn.Conv2d(24, 12, 3, padding="same"), nn.ReLU()
    )
    self.merge_layer = nn.Conv2d(in_channels=30, out_channels=1, kernel_size=1)
    self.initialize_weights()

  def forward(self, x):
    x1 = self.branch1(x)
    x2 = self.branch2(x)
    x3 = self.branch3(x)
    x = torch.cat((x1, x2, x3), dim=1)
    # 不是分类问题，最后一层不需要激活函数
    x = self.merge_layer(x)
    return x

  # pytorch内置的那些module(layer)都已经参数化好了，即使不人为初始化参数也可以进行训练
  def initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)

        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

      elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
