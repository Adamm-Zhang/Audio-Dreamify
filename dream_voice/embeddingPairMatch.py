from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.neighbors import NearestNeighbors
from utils import flatten_embedding
import numpy as np
import librosa
import openl3
from pathlib import Path
import dream_voice
import joblib
import pandas as pd
import torch

def load_audio_stereo(path, sr=44100):
    # librosa returns (2, N) if mono=False
    y, sr = librosa.load(path, sr=sr, mono=False)
    
    # Ensure shape is (2, N) for stereo; if mono file, make it (1, N)
    if y.ndim == 1:
        y = y[None, :]
    return y, sr

# def openl3_embed(
#     y: np.ndarray,
#     sr: int = 44100,
#     embedding_size: int = 512,
#     content_type: str = "music",
#     input_repr: str = "mel256",
#     )->tuple[np.ndarray, np.ndarray]:
    
#     # ts is time stamps, emb_seq is the sequence of embeddings
#     emb_seq, ts = openl3.get_audio_embedding(
#         y, sr,
#         content_type=content_type,
#         input_repr=input_repr,
#         embedding_size=embedding_size,
#         hop_size=0.1
#     )
#     emb_mean = emb_seq.mean(axis=0)
#     # L2-normalize for cosine similarity matching
#     emb_mean = emb_mean / (np.linalg.norm(emb_mean) + 1e-12)
#     return emb_seq, emb_mean

# manual model load
# https://acids-ircam.github.io/rave_models_download
model_path = "./dream_voice/musicnet.ts" 
model = torch.jit.load(model_path)
model.eval()


def get_invertible_embedding(audio_path, rave_model):
    x, sr = librosa.load(audio_path, sr=44100, mono=True)
    
    # Prepare tensor: (Batch, Channels, Time)
    x_tensor = torch.from_numpy(x).reshape(1, 1, -1).float()
    
    # Encode to latent space 'z'
    # z shape is typically (Batch, Latent_Dim, Time_Frames)
    with torch.no_grad():
        z = rave_model.encode(x_tensor)
        
    return z


@dataclass
class Segment:
    path: str
    duration_s: float
    features: np.ndarray          # (F,)
    cluster_id: int               # 0..K-1
    emb_mean: np.ndarray          # (512,) L2-normalized


# y, sr = load_audio_stereo(r"./dream_voice/dreamSegments/conversations_segment_4.mp3", sr=44100)
# seq, emb_mean = openl3_embed(
#     y, sr,
#     content_type="music",
#     input_repr="mel256",
#     embedding_size=512 
# )
# print(seq.shape)
# print(emb_mean.shape)  # Example usage

# for data of each segment from a directory of segments, build Segment dataclass objects
def build_segments(segment_items, kmeans, rave_model) -> List[Segment]:
    """
    segment_items: iterable of dicts like:
      {"path":..., "start_s":..., "duration_s":..., "features": np.array(F,)}
    """
    out = []
    
    for item in segment_items:
        # print(type(item['features']))
        cid = kmeans.predict(pd.DataFrame([item["features"]]))[0]
        print(cid)

        # item["path"] is an already sliced 5 second clip
        embedding = get_invertible_embedding(item["path"], rave_model)
        # print(embedding.shape)
        print("created embeddings for:", item["path"])
        
        out.append(Segment(
            path=item["path"],
            duration_s=item.get("duration_s", 0.0),
            features=item["features"],
            cluster_id=cid,
            emb_mean=embedding
        ))

    min_time = min([x.emb_mean.shape[-1] for x in out])
    for segment in out:
        segment.emb_mean = segment.emb_mean[:, :, :min_time]

    return out

def truncate_segment_lists(segmentList1: List[Segment], segmentList2: List[Segment]) -> Tuple[List[Segment], List[Segment]]:
    min_time = min([x.emb_mean.shape[-1] for x in segmentList1 + segmentList2])
    for segment in segmentList1:
        segment.emb_mean = segment.emb_mean[:, :, :min_time]
        print(segment.emb_mean.shape)
    for segment in segmentList2:
        segment.emb_mean = segment.emb_mean[:, :, :min_time]
        print(segment.emb_mean.shape)
    return segmentList1, segmentList2

# dreamSongs = Path(r"./dream_voice/dreamSegments")
# trapSongs = Path(r"./dream_voice/trapSegments")
# DSP_process = dream_voice.dreamSectionClassifier()

# pretrained kmeans model
section_classifier = joblib.load("kmeans_section_classifier.joblib")

# collect segment data for all segments in a folder; each segment has metadata + dsp data in a dictionary
# so we can get kmeans section cluster and openl3 embeddings later
def collect_seg_data(path: Path, DSP_process) -> List[Dict]:
    segments = []
    for file in path.glob("*.mp3"):
        features = DSP_process.fullFeatureExtract(str(file))
        duration = librosa.get_duration(filename=str(file))
        
        segment_items = {
            "path": str(file),
            "duration_s": duration,
            "features": features
        }
        # print(f"segment items: {segment_items}")
        segments.append(segment_items)
    return segments


def pair_segments_by_cluster_nn(
    trap_segments: List[Segment],
    dream_segments: List[Segment],
    k: int = 1,) -> List[Tuple[Segment, Segment, float]]:
    """
    Returns list of (trap_seg, dream_seg, cosine_distance)
    Uses nearest neighbors within each cluster.
    """
    # Group dream segments by cluster
    dream_by_cluster: Dict[int, List[Segment]] = {}
    for s in dream_segments:
        dream_by_cluster.setdefault(s.cluster_id, []).append(s)

    # dream_segments_by_cluster = dict(id: List[Segment])
    
    pairs = []
    print(sorted(set(s.cluster_id for s in trap_segments)))
    
    for cid in sorted(set(s.cluster_id for s in trap_segments)):    # for all cluster id's that exist
        trap_c = [s for s in trap_segments if s.cluster_id == cid]  # trap_c = list(segment) for current cluster id; all segments
        dream_c = dream_by_cluster.get(cid, [])                     # dream_c = list(segment) for current cluster id; all segments
        if len(trap_c) == 0 or len(dream_c) == 0:
            continue

        X_dream = flatten_embedding(dream_c)    # (Nd, 16)
        nn = NearestNeighbors(n_neighbors=min(k, len(dream_c)), metric="cosine")
        nn.fit(X_dream)

        X_trap = flatten_embedding(trap_c)      # (Nt, 16)
        dists, idxs = nn.kneighbors(X_trap, return_distance=True)

        print(idxs.shape)
        # Choose the best match (k=1) or store top-k
        for i, trap_seg in enumerate(trap_c):
            j = int(idxs[i][0])
            dist = float(dists[i][0])
            pairs.append((trap_seg, dream_c[j], dist))

    # Optional: sort by best matches first
    pairs.sort(key=lambda t: t[2])
    return pairs

def make_pt_file(pairList, outputTrapPath, outputDreamPath):
    Xtrap = torch.tensor([])
    Xdream = torch.tensor([])
    for i, (trap_seg, dream_seg, dist) in enumerate(pairList):
       Xtrap = torch.cat((Xtrap, trap_seg.emb_mean), 0)
       Xdream = torch.cat((Xdream, dream_seg.emb_mean), 0)
    
    print(f"XTRAP SHAPE: {Xtrap.shape}")
    print(f"XDREAM SHAPE: {Xdream.shape}")
    torch.save(Xtrap, outputTrapPath)
    torch.save(Xdream, outputDreamPath)


def main_get_embedding_pairs(rave_model, section_classifier, dreamSegments, trapSegments, DSP_process):
# refactor to collect functions into segment builder class    
    segments_dream = collect_seg_data(dreamSegments, DSP_process)
    segment_objs_dream = build_segments(segments_dream, section_classifier, rave_model=rave_model)
    print(type(segment_objs_dream))
    print(f"Built {len(segment_objs_dream)} dream segment objects.")


    segments_trap = collect_seg_data(trapSegments, DSP_process)
    segment_objs_trap = build_segments(segments_trap, section_classifier, rave_model=rave_model)
    print(type(segment_objs_trap))
    print(f"Built {len(segment_objs_trap)} trap segment objects.")

    segment_objs_dream, segment_objs_trap = truncate_segment_lists(segment_objs_dream, segment_objs_trap)

    # there's 1 pair being formed with trap and dream because theres only cluster 0 and 2 in dream, while trap has a single 0 cluster
    # must rework dataset and cluster generation
    # algorithm works though
    pairs = pair_segments_by_cluster_nn(segment_objs_dream, segment_objs_dream, k=1)
    print(f"Found {len(pairs)} pairs.")
    make_pt_file(pairs, r"./dream_voice/training_data_pt/embeddingPairsTrap.pt", r"./dream_voice/training_data_pt/embeddingPairsDream.pt")

'''
dreamSegments = Path(r"./dream_voice/dreamSegments")
trapSegments = Path(r"./dream_voice/trapSegments")

# need this to classify segments in our dataset
DSP_process = dream_voice.dreamSectionClassifier()
main_get_embedding_pairs(model, section_classifier, dreamSegments, trapSegments, DSP_process)
'''