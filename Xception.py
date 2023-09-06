# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:36:49 2023

@author: cakir
"""

import torch.nn as nn
from torch.nn import functional as F

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Xception(nn.Module):
  def __init__(self, num_classes = 1000):
    super(Xception, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 2, bias = False)
    self.batchNorm = nn.BatchNorm2d(32)
    self.activationRelu = nn.ReLU()
    self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, bias = False)
    self.batchNorm2 = nn.BatchNorm2d(64)

    self.bconv1 = self._make_bconv(64, 128)
    self.block1 = self._make_block(64, 128, relu_active = False)

    self.bconv2 = self._make_bconv(128, 256)
    self.block2 = self._make_block(128, 256)

    self.bconv3 = self._make_bconv(256, 728)
    self.block3 = self._make_block(256, 728)

    self.middleBlock = self._make_block(728, 728, max_active=False, is_middleFlow=True)

    self.bconv4 = self._make_bconv(728, 1024)
    self.block4 = self._make_block(728, 728, is_exitFlow=True, exitflow_planes=1024)

    self.exitSep1 = SeparableConv2d(1024, 1536, 3, padding = 1)
    self.exitSep1BN = nn.BatchNorm2d(1536)

    self.exitSep2 = SeparableConv2d(1536, 2048, 3, padding = 1)
    self.exitSep2BN = nn.BatchNorm2d(2048)

    self.globalAvgPool = nn.AdaptiveAvgPool2d(())

    self.fc = nn.Linear(2048, num_classes)

  def _make_block(self, in_planes, out_planes, relu_active = True, max_active = True,is_middleFlow = False, is_exitFlow = False, exitflow_planes = 0):
    layers = []
    if relu_active:
      layers.append(nn.ReLU())

    if is_exitFlow:
      exit_planes = exitflow_planes
    else:
      exit_planes = out_planes

    block = nn.Sequential(
        SeparableConv2d(in_planes, out_planes, 3, padding = 1),
        nn.BatchNorm2d(out_planes),

        nn.ReLU(),
        SeparableConv2d(out_planes, exit_planes, 3, padding = 1),
        nn.BatchNorm2d(exit_planes),
      )

    layers.append(block)

    if is_middleFlow:
      middle_part = nn.Sequential(
        nn.ReLU(),
        SeparableConv2d(out_planes, exit_planes, 3, padding = 1),
        nn.BatchNorm2d(exit_planes),                         
      )
      layers.append(middle_part)

    if max_active:
      layers.append(nn.MaxPool2d(3, 2, padding = 1))

    return nn.Sequential(*layers)

  def _make_bconv(self, in_planes, out_planes):
    layers = []
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = 2, bias = False))
    layers.append(nn.BatchNorm2d(out_planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.batchNorm(x)
    x = self.activationRelu(x)


    x = self.conv2(x)
    x = self.batchNorm2(x)
    x = self.activationRelu(x)

    
# ==============  Entry Flow   ================
    for block, bconv in zip([self.block1, self.block2, self.block3],
                             [self.bconv1, self.bconv2, self.bconv3]):
        prev_x = x
        x = block(x)
        prev_x = bconv(prev_x)
        x += prev_x
        
# ============== Middle Flow ===================
    prev_x = x
    for _ in range(8):
      x = self.middleBlock(x)
    x += prev_x

# ============== Exit Flow ===================
    prev_x = x
    x = self.block4(x)
    prev_x = self.bconv4(prev_x)
    x += prev_x
    
    x = self.exitSep1(x)
    x = self.exitSep1BN(x)
    x = self.activationRelu(x)


    x = self.exitSep2(x)
    x = self.exitSep2BN(x)
    x = self.activationRelu(x)

# ========== Fully Connected Layer ========
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = F.softmax(x, dim = 1)

    return x
