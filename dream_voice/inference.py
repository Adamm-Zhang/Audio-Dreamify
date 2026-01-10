import torch
import librosa
import soundfile as sf
import os
from resnet import ResNet1D

class DreamVoicePredictor:
    def __init__(self, rave_path, mapper_path, device='cuda', sr=44100):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.rave = torch.jit.load(rave_path).to(self.device)
        self.rave.eval()

        self.dreamMap = ResNet1D(init_dim=16, hidden_dim=64)
        self.dreamMap.load_state_dict(torch.load(mapper_path, map_location=self.device))
        self.dreamMap.to(self.device)
        self.dreamMap.eval()
        
        self.sr = sr

    def preprocess_audio_librosa(self, audio_path, target_sr=48000):
      # need mono for rave embedding
      audio_np, sr = librosa.load(audio_path, sr=target_sr, mono=True)
      y = torch.from_numpy(audio_np).float()
      
      # Librosa returns (Time,), but RAVE expects (Batch, Channels, Time) -> (1, 1, T)
      y = y.unsqueeze(0).unsqueeze(0)
      return y.to(self.device)

    def predict(self, input_path, output_path, mix_ratio = 1.0):
        print(f"\nProcessing: {input_path}")
        
        x_trap = self.preprocess_audio_librosa(input_path)
        
        with torch.no_grad():
            z_trap = self.rave.encode(x_trap).to(self.device)
            z_dreamy = self.dreamMap(z_trap)
            
            if mix_ratio == 1.0:
                z_final = self.dreamMap(z_trap)
            elif mix_ratio == 0.0:
                z_final = z_trap
            else:
                z_final = (z_trap * (1 - mix_ratio)) + (z_dreamy * mix_ratio)
            
            print("decoding rave embedding to audio")
            y_dreamy = self.rave.decode(z_final)


        # remove batch dim
        y_dreamy = y_dreamy.squeeze(0).cpu()
        
        # Check for clipping (optional)
        if torch.max(torch.abs(y_dreamy)) > 1.0:
            print("   ⚠️ Output clipped! Normalizing volume...")
            y_dreamy = y_dreamy / torch.max(torch.abs(y_dreamy))

        audio_np = y_dreamy.detach().cpu().numpy()
        audio_np = audio_np.T
        sf.write(output_path, audio_np, samplerate=self.sr)
        print(f"✅ Saved to: {output_path}")
        
if __name__ == "__main__":
    MIX = 0.3
    rave_path = "./dream_voice/musicnet.ts"
    dreamify_map_path = "./dream_voice/completedModels/dreamify.pth"
  
    TEST_SEGMENT = "./dream_voice/trapSegments/The Chainsmokers - Summertime Friends (Official Video)_segment_18.mp3"
    output_path = "./dream_voice/output_tests/testseg2.wav"
    
    dream = DreamVoicePredictor(rave_path, dreamify_map_path)
    dream.predict(TEST_SEGMENT, output_path, MIX)
  
  