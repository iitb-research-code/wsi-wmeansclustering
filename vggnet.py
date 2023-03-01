# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
import numpy as np


d_type     = 'd1'     # distance type



class VGGNet(VGG):
  def __init__(self, pretrained=True, model='vgg16', requires_grad=False, remove_fc=False, show_params=False):
    super().__init__(make_layers(cfg[model]))
    self.ranges = ranges[model]
    self.fc_ranges = ((0, 2), (2, 5), (5, 7))

    if pretrained:
      exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

    if not requires_grad:
      for param in super().parameters():
        param.requires_grad = False

    if remove_fc:  # delete redundant fully-connected layer params, can save memory
      del self.classifier

    if show_params:
      for name, param in self.named_parameters():
        print(name, param.size())

  def forward(self, x):
    output = {}

    x = self.features(x)

    avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
    avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
    avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
    output['avg'] = avg

    x = x.view(x.size(0), -1)  # flatten()
    dims = x.size(1)
    if dims >= 25088:
      x = x[:, :25088]
      for idx in range(len(self.fc_ranges)):
        for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
          x = self.classifier[layer](x)
        output["fc%d"%(idx+1)] = x
    else:
      w = self.classifier[0].weight[:, :dims]
      b = self.classifier[0].bias
      x = torch.matmul(x, w.t()) + b
      x = self.classifier[1](x)
      output["fc1"] = x
      for idx in range(1, len(self.fc_ranges)):
        for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
          x = self.classifier[layer](x)
        output["fc%d"%(idx+1)] = x

    return output
  

ranges = {
  'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
  'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
  'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
  'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
  'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
  layers = []
  in_channels = 3
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)