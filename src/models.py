import torch
from torch import nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights, get_weight


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


class CSRNet(nn.Module):
  def __init__(self, config: int = 3, freeze_frontend=True):
    super().__init__()
    vgg16 = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    # frontend
    self.frontend: nn.Module = vgg16.features[:33]  # 保留0-32层
    self.backend_config = [512, 512, 512, 256, 128, 64]
    if config == 1:
      self.backend_dilations = [1, 1, 1, 1, 1, 1]
    elif config == 2:
      self.backend_dilations = [2, 2, 2, 2, 2, 2]
    elif config == 3:
      self.backend_dilations = [2, 2, 2, 4, 4, 4]
    else:
      self.backend_dilations = [4, 4, 4, 4, 4, 4]
    # backend
    self.backend: nn.Module = self.make_layers(
        self.backend_config, self.backend_dilations, in_channels=512, batch_norm=True)
    # output
    self.output_layer = nn.Conv2d(
        in_channels=64, out_channels=1, kernel_size=1)
    self.initialize_weights(self.backend)
    self.initialize_weights(self.output_layer)
    if freeze_frontend:
      for param in self.frontend.parameters():
        param.requires_grad = False

  def forward(self, x):
    x = self.frontend(x)
    x = self.backend(x)
    x = self.output_layer(x)
    return x

  # pytorch内置的那些module(layer)都已经参数化好了，即使不人为初始化参数也可以进行训练
  @staticmethod
  def initialize_weights(model: nn.Module):
    for m in model.modules():
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

  @staticmethod
  def make_layers(cfg: list, dilations: list, in_channels=512, batch_norm=False) -> nn.Module:
    layers = []
    for i, v in enumerate(cfg):
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                         padding=dilations[i], dilation=dilations[i])
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
    return nn.Sequential(*layers)
