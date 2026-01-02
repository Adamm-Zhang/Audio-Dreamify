import ffmpeg
import librosa
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import utils
from separation_data_generation import segmentGenerator
from pathlib import Path
import pandas as pd
import sklearn
import pickle
import joblib

class StereoMixin:
    def midSideDecompose(self, y):
        mid = (y[0] + y[1]) / 2
        side = (y[0] - y[1]) / 2
        return {'mid': mid, 'side': side}

class sectionClassifier(ABC, StereoMixin):
    def __init__(self, sr=44100, hop_length=512, n_fft=2048):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
    
    @abstractmethod
    def fullFeatureExtract(self, audio_path):
        pass

    def extract_base_features(self, y, sr):
        mid = self.midSideDecompose(y)['mid']
        side = self.midSideDecompose(y)['side']
        
        # stereoWidth = np.mean(np.abs(side)) / (np.mean(np.abs(mid)) + 1e-6)
        rms_M = librosa.feature.rms(y=mid).mean()
        rms_S = librosa.feature.rms(y=side).mean()
        
        M_dB = 20 * np.log10(rms_M + 1e-9)
        S_dB = 20 * np.log10(rms_S + 1e-9)
        # Stereo width ratio (dB difference)
        MS_ratio = S_dB - M_dB
        
        rms_val = np.sqrt(np.mean((y[0]**2 + y[1]**2) / 2))
        avg_dynamic_range = utils.monoDynamicRange(mid)
        
        spectral_centroid_Mid = librosa.feature.spectral_centroid(y=mid, sr=sr).mean()
        spectral_centroid_Side = librosa.feature.spectral_centroid(y=side, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y=mid).mean()
        flatness = librosa.feature.spectral_flatness(y=mid).mean()
        
        return {
            'rms': rms_val,
            'spectral_centroid_mid': spectral_centroid_Mid,
            'spectral_centroid_side': spectral_centroid_Side,
            'flatness': flatness,
            'stereo ratio': MS_ratio,
            'avg_dynamic_range': avg_dynamic_range**2
        }
        
    def getBands(self, y, lowCut, highCut):
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, window='hann'))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        # print(stft.shape)
        
        # these are spectrographs
        low_band = stft[(freqs >= 25) & (freqs <= lowCut)].mean()
        high_band = stft[(freqs >= highCut) & (freqs <= 20000)].mean()
        mid_band = stft[(freqs > lowCut) & (freqs < highCut)].mean()
        
        return low_band, mid_band, high_band

class dreamSectionClassifier(sectionClassifier):
    def __init__(self, low_band=300, high_band=3000):
        super().__init__()
        self.low_band = low_band
        self.high_band = high_band
        print(self.sr)
        
    def fullFeatureExtract(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr, mono=False)
        #print(y.shape)
        features = self.extract_base_features(y, sr=self.sr)
        
        y_mid = self.midSideDecompose(y)['mid']
        # average low band energy
        low_band, mid_band, high_band = self.getBands(y_mid, lowCut=self.low_band, highCut=self.high_band)

        features['low_band'] = low_band
        features['mid_band'] = mid_band
        features['high_band'] = high_band
        
        return features

class kmeansSectionClassifier():
    def __init__(self, n_clusters=3):
        self.classifier = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.trained = False
        
    def fit(self, feature_dataframe):
        scaled_features = self.scaler.fit_transform(feature_dataframe)
        self.classifier.fit(scaled_features)
        self.trained = True
        
    def predict(self, feature_dataframe):
        if not self.trained:
            raise Exception("Classifier not trained. Call fit() first.")
        scaled_features = self.scaler.transform(feature_dataframe)
        return self.classifier.predict(scaled_features)

def splitSongs(dreamSongs, trapSongs, dreamOutputDirect, trapOutputDirect):
    for file in dreamSongs.glob("*.mp3"):
        print("Processing file:", file)
        seg_gen = segmentGenerator(file)
        seg_gen.generate_and_save_segments(dreamOutputDirect, file.stem)
    
    for file in trapSongs.glob("*.mp3"):
        print("Processing file:", file)
        seg_gen = segmentGenerator(file)
        seg_gen.generate_and_save_segments(trapOutputDirect, file.stem)
    

classifier1 = dreamSectionClassifier()
# features = classifier1.fullFeatureExtract(r"./dream_voice/segment_0.mp3")

# directory_path = Path(r"./dream_voice/audioFiles")
dreamSongs = Path(r"./dream_voice/fullDreamSongs")
trapSongs = Path(r"./dream_voice/fullTrapSongs")
dreamSegmentsOutput = Path(r"./dream_voice/dreamSegments")
trapSegmentsOutput = Path(r"./dream_voice/trapSegments")

# reformat these - copy code

for file in dreamSongs.glob("*.mp3"):
    print("Processing file:", file)
    seg_gen = segmentGenerator(file)
    seg_gen.generate_and_save_segments(r"./dream_voice/dreamSegments", file.stem)

for file in trapSongs.glob("*.mp3"):
    seg_gen = segmentGenerator(file)
    seg_gen.generate_and_save_segments(r"./dream_voice/trapSegments", file.stem)

kmeans_dream = kmeansSectionClassifier(n_clusters=3)
kmeansDataframe = pd.DataFrame()
fileNames = []

for directory in [dreamSegmentsOutput, trapSegmentsOutput]:
    for file in directory.glob("*.mp3"):
        features = classifier1.fullFeatureExtract(str(file))
        kmeansDataframe = kmeansDataframe._append(features, ignore_index=True)
        fileNames.append(file.name)


kmeans_dream.fit(kmeansDataframe)

kmeansDataframe['Cluster'] = kmeans_dream.classifier.labels_
kmeansDataframe['fileName'] = fileNames

print(kmeansDataframe)

joblib.dump(kmeans_dream, "kmeans_section_classifier.joblib")