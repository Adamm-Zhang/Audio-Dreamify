import utils

class segmentGenerator:
    def __init__(self, audio_path, segment_duration=5.0, sr=44100):
        self.audio_path = audio_path
        self.segment_duration = segment_duration
        self.sr = sr

    def generate_and_save_segments(self, output_folder, songName):
        segments = utils.splitAudio(self.audio_path, self.segment_duration)
        for i, segment in enumerate(segments):
            if utils.isGoodSegment(segment, self.sr):                
                if i%2 == 0:
                    output_path = f"{output_folder}/{songName}_segment_{i}.mp3"
                    utils.saveAudioSegment(segment, self.sr, output_path)
                    print(f"Saved good segment: {output_path}")
                
                
