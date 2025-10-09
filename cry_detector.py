import torch
import torchaudio
import torch.nn.functional as F
from panns_inference import AudioTagging, labels

class CryDetector:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CryDetector using device: {self.device}")
        self.at = AudioTagging(device=self.device)
        self.labels = labels  # list of 527 AudioSet class names
        self.target_labels = [
            "Crying, sobbing",
            "Baby cry, infant cry",
            "Whimper"
        ]

    def preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 32000:
            waveform = torchaudio.functional.resample(waveform, sr, 32000)
        waveform = waveform.mean(dim=0)  # mono, shape (N)
        return waveform.unsqueeze(0)  # shape (1, 1, N)

    def detect_cry(self, audio_path, threshold=0.1):
        wav = self.preprocess(audio_path).to(self.device)
        # AudioTagging expects shape (batch, samples)
        clipwise_output, embedding = self.at.inference(wav)
        # clipwise_output shape: (batch, 527)
        probs = torch.tensor(clipwise_output[0])

        cry_score = 0.0
        for t in self.target_labels:
            if t in self.labels:
                idx = self.labels.index(t)
                cry_score += float(probs[idx])

        cry_score = min(cry_score, 1.0)
        is_cry = cry_score >= threshold

        return {"cry_confidence": round(cry_score, 3), "is_cry": is_cry}

if __name__ == "__main__":
    import sys
    detector = CryDetector()
    result = detector.detect_cry(sys.argv[1])
    print(result)