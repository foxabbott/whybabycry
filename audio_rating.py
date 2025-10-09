import torch
import torchaudio
import numpy as np
from transformers import pipeline
from cry_detector import CryDetector
import argparse
import json
import sys

class AudioCriteriaRater:
    def __init__(self, model_name="laion/clap-htsat-fused", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device set to use {self.device}")
        self.classifier = pipeline(
            task="zero-shot-audio-classification",
            model=model_name,
            device=0 if self.device != "cpu" else -1
        )
        self.baseline_prompt = "A quiet room with no crying or human sounds"

    def preprocess_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
        return waveform, 16000

    def compute_similarity(self, audio_path, criteria):
        waveform, sr = self.preprocess_audio(audio_path)

        # Get baseline similarity (quiet room)
        baseline_result = self.classifier(
            audios=waveform.squeeze().numpy(),
            candidate_labels=[self.baseline_prompt]
        )
        baseline_sim = baseline_result[0]['score']

        # Get similarities for all criteria
        results = self.classifier(audios=waveform.squeeze().numpy(), candidate_labels=criteria)
        raw_scores = np.array([r['score'] for r in results])

        # Contrastive normalization (subtract baseline)
        contrastive = raw_scores - baseline_sim

        # Independent scaling (map to 1–10)
        min_s, max_s = contrastive.min(), contrastive.max()
        scaled = 1 + 9 * (contrastive - min_s) / (max_s - min_s + 1e-9)

        return list(zip(criteria, scaled.tolist()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Rate audio against criteria with cry detection and CLAP analysis.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("--criteria_file", type=str, default="default_criteria.json", help="JSON file with criteria list")
    parser.add_argument("--threshold", type=float, default=0.3, help="Cry detection threshold")
    args = parser.parse_args()

    # Stage 1: Cry detection
    cry_detector = CryDetector()
    cry_result = cry_detector.detect_cry(args.audio_path, threshold=args.threshold)

    if not cry_result["is_cry"]:
        print(f"No baby cry detected (confidence = {cry_result['cry_confidence']:.2f})")
        sys.exit(0)
    else:
        print(f"Baby cry detected (confidence = {cry_result['cry_confidence']:.2f})")

    # Stage 2: CLAP-based scoring
    with open(args.criteria_file, "r") as f:
        data = json.load(f)
        criteria = data.get("criteria", [])

    rater = AudioCriteriaRater()
    results = rater.compute_similarity(args.audio_path, criteria)

    for c, s in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{c}: {s:.2f}/10")