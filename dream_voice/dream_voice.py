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
import os
import torch

### Main purpose of this file: ###
# initial datapipeline from full songs to usable 5s segments for resnet model
# extract general DSP features
# create kmeans model to automatically classify segments by approximate song section; likely verse, chorus, bridge
# basically to ensure main resnet model trains off apple-to-apple pairs

# this runs separately from resnet main for now!

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
        
        flux_low, flux_mid, flux_high = utils.calc_spectral_flux(torch.from_numpy(mid))
        return {
            'rms': rms_val,
            'spectral_centroid_mid': spectral_centroid_Mid,
            'spectral_centroid_side': spectral_centroid_Side,
            'flatness': flatness,
            'stereo ratio': MS_ratio,
            'avg_dynamic_range': avg_dynamic_range**2,
            'L spectral flux': flux_low,
            'M spectral flux': flux_mid,
            'H spectral flux': flux_high,
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
    def __init__(self, feature_names, n_clusters=3):
        self.classifier = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.trained = False
        self.feature_names = feature_names
        
    def fit(self, feature_dataframe):
        scaled_features = self.scaler.fit_transform(feature_dataframe)
        self.classifier.fit(scaled_features)
        self.trained = True
        
    def predict(self, feature_dataframe):
        if not self.trained:
            raise Exception("Classifier not trained. Call fit() first.")
        scaled_features = self.scaler.transform(feature_dataframe)
        return self.classifier.predict(scaled_features)
    
    # property decorator instead of static variable; we need to get cluster centers AFTER we fit; 
    # property makes python recalculate this value everytime we ask for it
    # more robust
    # NOTE: @property automatically registers this as a getter function; calling classifier.get_cluster_centers invokes the function
    @property
    def get_cluster_centers(self):
        return self.classifier.cluster_centers_
    
    @property   # this is getter only; no public writes
    def get_feature_names_in(self):
        return self.feature_names

def splitSongs(dreamSongs, trapSongs, dreamOutputDirect, trapOutputDirect):
    for file in dreamSongs.glob("*.mp3"):
        print("Processing file:", file)
        seg_gen = segmentGenerator(file)
        seg_gen.generate_and_save_segments(dreamOutputDirect, file.stem)
    
    for file in trapSongs.glob("*.mp3"):
        print("Processing file:", file)
        seg_gen = segmentGenerator(file)
        seg_gen.generate_and_save_segments(trapOutputDirect, file.stem)
    
if __name__ == "__main__":
    # only need 1 classifier object; classification parameters are the same
    classifier1 = dreamSectionClassifier()

    # features = classifier1.fullFeatureExtract(r"./dream_voice/segment_0.mp3")

    # directory_path = Path(r"./dream_voice/audioFiles")
    dreamSongs = Path(r".\dream_voice\songScrape\rawData\fullDreamSongs")
    trapSongs = Path(r".\dream_voice\songScrape\rawData\fullTrapSongs")
    dreamSegmentsOutput = Path(r"./dream_voice/dreamSegments")
    trapSegmentsOutput = Path(r"./dream_voice/trapSegments")

    os.makedirs(dreamSegmentsOutput, exist_ok=True)
    os.makedirs(trapSegmentsOutput, exist_ok=True)

    # reformat these - copy code

    for file in dreamSongs.glob("*.wav"):
        print("Processing file:", file)
        seg_gen = segmentGenerator(file)
        seg_gen.generate_and_save_segments(dreamSegmentsOutput, file.stem)

    for file in trapSongs.glob("*.wav"):
        seg_gen = segmentGenerator(file)
        seg_gen.generate_and_save_segments(trapSegmentsOutput, file.stem)


    # init dataframes to train kmeans on for DSP section classification
    dream_kmeansDataframe = pd.DataFrame()
    trap_kmeansDataframe = pd.DataFrame()

    # keep segment file name for tracking and validation later
    dream_fileNames = []
    trap_fileNames = []

    ###### dream ######

    # fill dream dataframe for kmeans training
    for file in dreamSegmentsOutput.glob("*.mp3"):
        features = classifier1.fullFeatureExtract(str(file))
        dream_kmeansDataframe = pd.concat([dream_kmeansDataframe, pd.DataFrame([features])], ignore_index=True)
        dream_fileNames.append(file.name)

    # need 2 classifiers; else we might overfit to some genre-specific trait
    # i.e. trap is louder in general; might match all trap segments to only dream choruses
    # we can see this with previous experimental results
    # NOTE: we define trap and dream classifier objects right before training so we can store feature_names in 1 go. 
    # using standard_scaler converts df to numpy array; can't access feature names later on for energy classification in utils
    kmeans_dream = kmeansSectionClassifier(feature_names=dream_kmeansDataframe.columns, n_clusters=3)
    kmeans_dream.fit(dream_kmeansDataframe)

    dream_kmeansDataframe['Cluster'] = kmeans_dream.classifier.labels_
    dream_kmeansDataframe['fileName'] = dream_fileNames

    ###### trap ######

    # fill trap dataframe for kmeans training
    for file in trapSegmentsOutput.glob("*.mp3"):
        features = classifier1.fullFeatureExtract(str(file))
        trap_kmeansDataframe = pd.concat([trap_kmeansDataframe, pd.DataFrame([features])], ignore_index=True)
        trap_fileNames.append(file.name)

    kmeans_trap = kmeansSectionClassifier(feature_names=trap_kmeansDataframe.columns, n_clusters=3)
    kmeans_trap.fit(trap_kmeansDataframe)

    trap_kmeansDataframe['Cluster'] = kmeans_trap.classifier.labels_
    trap_kmeansDataframe['fileName'] = trap_fileNames

    print(dream_kmeansDataframe)
    print(trap_kmeansDataframe)

    classifiersPath = r"./dream_voice/kmeans_classifiers"
    os.makedirs(classifiersPath, exist_ok=True)
    joblib.dump(kmeans_dream, os.path.join(classifiersPath, "dream_kmeans_section_classifier.joblib"))
    joblib.dump(kmeans_trap, os.path.join(classifiersPath, "trap_kmeans_section_classifier.joblib"))