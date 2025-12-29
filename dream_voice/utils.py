import numpy as np
import soundfile as sf
import math
import librosa

def splitAudio(audio_path, segment_duration=5.0):
    y, sr = sf.read(audio_path)
    # print(y.shape)
    total_duration = len(y) / sr
    segments = []
    
    for start in np.arange(0, total_duration, segment_duration):
        end = min(start + segment_duration, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segments.append(y[start_sample:end_sample])
    
    return segments

def saveAudioSegment(segment, sr, output_path):
    sf.write(output_path, segment, sr)
     

def monoSignalRMS(y):   # y is a 1D numpy array with shape for mono audio
    rms_val = np.sqrt(y**2).mean()
    return rms_val

def monoDynamicRange(y):   # y is a 1D numpy array with shape for mono audio
    rms = librosa.feature.rms(y=y)
    rms_db = 20 * np.log10(rms + 1e-9)
    dynamic_range = np.max(rms_db) - np.mean(rms_db)
    return dynamic_range

def isGoodSegment(segment, sr=44100):    # segment is a numpy array
    # combine to mono
    segment = segment.T  # transpose to shape (channels, samples)
    first_second = (segment[0][:sr] + segment[1][:sr]) / 2
    last_second = (segment[0][-sr:] + segment[1][-sr:]) / 2
        
    rms_first = monoSignalRMS(first_second)
    rms_last = monoSignalRMS(last_second)
    dynamicRangeFirst = monoDynamicRange(first_second)
    dynamicRangeLast = monoDynamicRange(last_second)
    
    print(f"dynamic range: {abs(dynamicRangeFirst - dynamicRangeLast)}")
    # print(abs(rms_first - rms_last))
    return abs(rms_first - rms_last) <= 0.03 and rms_first >= 0.02 and rms_last >= 0.02 and abs(dynamicRangeFirst - dynamicRangeLast) <= 2
                

def flatten_embedding(clusterList):
    vectors = []

    print(f"Processing {len(clusterList)} Dreamy segments...")

    for s in clusterList:
        # 1. Get the Raw Embedding: Shape (16, 108) or similar
        # Ensure it's a numpy array first
        if hasattr(s.emb_mean, 'numpy'):
            raw_emb = s.emb_mean.numpy()
        else:
            raw_emb = s.emb_mean
            
        # Ensure shape is (Channels, Time) -> (16, 108)
        # If it is (1, 16, 108), squeeze the first dim
        if raw_emb.ndim == 3:
            raw_emb = raw_emb.squeeze(0)
            
        # 2. Calculate Statistics (The "Fingerprint")
        # Axis=1 (or -1) is the Time dimension
        mu = np.mean(raw_emb, axis=-1)  # Mean (Timbre) -> Shape (16,)
        std = np.std(raw_emb, axis=-1)  # Std  (Activity) -> Shape (16,)
        
        # 3. Concatenate to make a 32-dim vector
        fingerprint = np.concatenate([mu, std]) # Shape (32,)
        
        vectors.append(fingerprint)

    # 4. Stack for KNN
    X = np.stack(vectors, axis=0)
    print(f"✅ X_dream shape: {X.shape}") 
    # Expected: (Num_Samples, 32)
    
    return X