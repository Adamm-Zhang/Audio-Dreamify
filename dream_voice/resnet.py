import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import numpy as np
import os

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
    super().__init__()
    padding = (kernel_size - 1) // 2 * dilation # padding to maintain length
    self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    self.norm1 = nn.InstanceNorm1d(num_features=out_channels)
    self.activation = nn.SiLU()
    
    self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    self.norm2 = nn.InstanceNorm1d(num_features=out_channels)
    
    if in_channels != out_channels:
      # 1x1 convolution to match dimensions; manipulate input channels to out_channels
      print(f"Creating residual skip conv from {in_channels} to {out_channels}")
      self.residual_skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    else:
      # no downsampling; identity skip
      # dilation doesnt mess with resolution since padding is used
      self.residual_skip = nn.Identity()
        
  
  def forward(self, x):
    residual = self.residual_skip(x)
    
    sequence = nn.Sequential(self.conv1, self.norm1, self.activation,
                             self.conv2, self.norm2, self.activation)
    out = sequence(x)
    out = self.activation(out + residual)
        
    return out
  
# 16 channel to 64 channel
# skip 32 channel block for maximum thinking channels; don't need to entangle properties
# small latent input size; no need for gradual increase; operations on 64 channels is fine
# 16-64 is 4x expansion; less redunancy in latent space
# 128 might learn specific pattern rhythms even though pairs arent perfect
# receptive field (time context, i.e. rhythm) > channel width (feature complexity, i.e. texture)
# increase to 128 if underfitting observed
class ResNet1D(nn.Module):
  def __init__(self, init_dim=16, hidden_dim=64):
    super().__init__()
    
    self.layer1 = ResidualBlock(init_dim, hidden_dim, kernel_size=3, dilation=1)
    
    self.layer2 = ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
    self.layer3 = ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
    self.layer4 = ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=8)
    self.layer5 = ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
    
    # recompress
    self.layer6 = ResidualBlock(hidden_dim, init_dim, kernel_size=3, dilation=1)
    
    self.layerSequence = nn.Sequential(self.layer1, self.layer2, self.layer3,
                                       self.layer4, self.layer5, self.layer6)
  
  def forward(self, x):
    out = self.layerSequence(x)
    return out
    

model1 = ResNet1D(init_dim=16, hidden_dim=64)

input_tensor = torch.randn(1, 16, 108)  # (Batch, Channels, Time)
out = model1(input_tensor)
print(out.shape)  # should be (1, 16, 108)

class transientLoss(nn.Module):
  def __init__(self, rave_model):
    super().__init__()
    self.rave_model = rave_model

    # eval so we dont get random dropout/batchnorm effects on rave; treat musicnet.ts as fixed feature extractor
    rave_model.eval()
    
    # freeze gradients so we dont backprop through rave
    for param in self.rave_model.parameters():
      param.requires_grad = False
    
  def get_envelope(self, embedding):
    signal = self.rave_model.decode(embedding)
    
    # rave.compute_envelope() might not exist in torchscript file
    # might be too fast of an envelope sampling window
    # goal is to blur signal to get transient envelope
    #envelope = self.rave_model.compute_envelope(signal)
    rectified = torch.abs(signal)
    
    # dont add padding here; might add silence artifacts that confuse model
    envelope = F.avg_pool1d(rectified, kernel_size=512, stride=256)
    return envelope

  def forward(self, input_signal_prediction, target_signal):
    
    input_envelope = self.get_envelope(input_signal_prediction)
    
    # safety; redundant since we fixed rave earlier
    with torch.no_grad():
      target_envelope = self.get_envelope(target_signal)
    
    # avoid 0 pads as a result of pooling
    min_length = min(input_envelope.shape[-1], target_envelope.shape[-1])
    input_envelope = input_envelope[..., :min_length]
    target_envelope = target_envelope[..., :min_length]
    
    return F.mse_loss(input_envelope, target_envelope)
  
if __name__ == "__main__":
  pass