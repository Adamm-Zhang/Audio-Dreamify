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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_PATH = "./dream_voice/test_data/train_loop_test_X_1000.pt"
Y_PATH = "./dream_voice/test_data/train_loop_test_Y_1000.pt"
RAVE_PATH = "./dream_voice/musicnet.ts"
RAVE_MODEL = torch.jit.load(RAVE_PATH).to(DEVICE).eval()
MODEL_SAVE_PATH = "dreamify.pth"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4

def main():
  print("working on {DEVICE}")
  
  X = torch.load(X_PATH).to(DEVICE)
  Y = torch.load(Y_PATH).to(DEVICE)

  if X.ndim == 4: X = X.squeeze(1)
  if Y.ndim == 4: Y = Y.squeeze(1)

  #X = torch.randn(100, 16, 108).to(DEVICE)
  #Y = torch.randn(100, 16, 108).to(DEVICE)

  print(X.shape, Y.shape)
  # tensor dataset: eager loading. data is small enough to fit in memory; 16 parameter embeddings
  dataset = torch.utils.data.TensorDataset(X, Y)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
  print("dataloader good")
  
  transient_loss_fn = transientLoss(RAVE_MODEL)
  print(hasattr(transient_loss_fn.rave_model, 'reset'))

  general_loss_fn = nn.MSELoss()
  
  resnet_model = ResNet1D(init_dim=16, hidden_dim=64).to(DEVICE)
  print("loaded RESNET model")
  
  optimizer = optim.Adam(resnet_model.parameters(), lr=LR)
  
  resnet_model.train()
  for epoch in range(EPOCHS):
    epoch_loss_general = 0.0
    epoch_loss_transient = 0.0
    epoch_loss_total = 0.0
    
    if hasattr(transient_loss_fn.rave_model, 'reset'):
        transient_loss_fn.rave.reset()
    
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
      
      
      predicted_embedding_batch  = resnet_model(x_batch)
      general_loss = general_loss_fn(predicted_embedding_batch, y_batch)
      transient_loss = transient_loss_fn(predicted_embedding_batch, y_batch)

      # tune this weight later
      total_loss = general_loss + 0.5 * transient_loss
      
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
      
      epoch_loss_general += general_loss.item()
      epoch_loss_transient += transient_loss.item()
      epoch_loss_total += total_loss.item()
      
    # MSELoss; reduction='mean' by default - batch loop gives mean error per batch
    # get average loss per epoch; len(dataloader) gives number of batches
    if epoch % 5 == 0:
      print(f"Epoch {epoch+1}/{EPOCHS}, General Loss: {epoch_loss_general/len(dataloader):.6f}, Transient Loss: {epoch_loss_transient/len(dataloader):.6f}, Total Loss: {epoch_loss_total/len(dataloader):.6f}")

  print("Training complete.")

if __name__ == "__main__":
  main()