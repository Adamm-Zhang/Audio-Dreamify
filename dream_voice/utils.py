import numpy as np
import pandas as pd
import soundfile as sf
import math
import librosa
import torch
import torch.nn.functional as F
import joblib

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


# latent flux
def calc_spectral_flux(y, n_fft=2048, hop_length=512):
    """
    Calculates standard DSP Spectral Flux (Onset Strength).
    Requires raw audio input.
    """
    # 1. Compute STFT (Spectrogram)
    # shape: (Batch, Freq, Time)
    spec = torch.stft(y, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    magnitude = torch.abs(spec)
    
    # 2. Positive Difference (Only counting energy *increases*)
    # We ignore energy drops (decays) because rhythm is defined by attacks.
    diff = magnitude[..., 1:] - magnitude[..., :-1]
    positive_diff = F.relu(diff)
    
    # 3. Sum across all frequencies
    flux_curve = torch.mean(positive_diff, dim=1)
    print(f"TYPE OF FLUX CURVE: {type(flux_curve)}")
    print(flux_curve.shape)
    
    return torch.mean(flux_curve[:14]), torch.mean(flux_curve[14:140]), torch.mean(flux_curve[140:])


# this remaps random cluster ID to cluster IDs sorted by energy; so highest energy always corresponds to chorus for example
# this ensures we don't accidentally pair chorus with verse for example
# does light DSP feature analysis with existing features
# order doesn't really matter; model shouldn't care about section, only signal similarity
def get_energy_sorted_mapping(kmeans_model):
    """
    Returns a dictionary that maps Random Cluster IDs -> Sorted Energy IDs.
    0 = Lowest Energy, 2 = Highest Energy.
    """
    # 1. Get the Centroids (Shape: n_clusters, n_features)
    centers = kmeans_model.get_cluster_centers
    centers_df = pd.DataFrame(centers, columns=kmeans_model.get_feature_names_in)
    # 2. Calculate "Total Energy" score for each centroid
    # We simply sum the Flux/RMS features. 
    # Since your features are all positive (Energy), sum = total intensity.
    energy_parameters = ['L spectral flux', 'M spectral flux', 'H spectral flux', 'rms', 'low_band']
    energy_scores = centers_df[energy_parameters].sum(axis=1)
    
    # 3. Sort the indices based on Energy (Low -> High)
    # argsort returns the Old IDs in the order of their energy.
    # Example: If Cluster 2 is quietest, sorted_indices[0] will be 2.
    sorted_indices = np.argsort(energy_scores)
    
    # 4. Create the Map
    # We want a lookup table: {Old_ID: New_Sorted_ID}
    mapping = {}
    for new_id, old_id in enumerate(sorted_indices):
        mapping[old_id] = new_id
        
    for old, new in mapping.items():
        print(f"Old Label {old} -> New Label {new} (Energy: {energy_scores[old]:.2f})")
        
    return mapping


# we can just directly get the mapping off the model when we load, have function for convenience
# make sure to run any output through the mapping. 
def load_remapped_kmeans(model_path):
    # automap kmeans when loading to avoid the cluster label shift problem
    model = joblib.load(model_path)
    
    mapping = get_energy_sorted_mapping(model)
    
    print(f"Loaded {model_path}")
    print(f"Generated Map: {mapping} (Based on saved weights)")
    
    return model, mapping