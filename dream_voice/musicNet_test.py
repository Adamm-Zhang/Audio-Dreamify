import torch
import librosa

model_path = "./dream_voice/musicnet.ts" 

print(f"Loading RAVE model from {model_path}...")

model = torch.jit.load(model_path)
model.eval()

print("Model loaded successfully")

# mono or stereo
try:
    dummy = torch.randn(1, 2, 65536)
    z = model.encode(dummy)
    print(f"Model is STEREO. Latent shape: {z.shape}")
except RuntimeError:
    dummy = torch.randn(1, 1, 65536)
    z = model.encode(dummy)
    print(f"Model is MONO. Latent shape: {z.shape}")