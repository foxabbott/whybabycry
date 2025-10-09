import os
import json
from audio_rating import AudioCriteriaRater
from cry_detector import CryDetector

# Path to the root directory containing subdirectories (classes)
ROOT_DIR = "audio_files/donateacry"
CRITERIA_FILE = "default_criteria.json"
CRY_THRESHOLD = 0.1  # or adjust as needed

# Load criteria
with open(CRITERIA_FILE, "r") as f:
    data = json.load(f)
    criteria = data.get("criteria", [])

rater = AudioCriteriaRater()
cry_detector = CryDetector()

results = []

for class_name in os.listdir(ROOT_DIR):
    class_dir = os.path.join(ROOT_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(class_dir, fname)
        # Cry detection
        cry_result = cry_detector.detect_cry(wav_path, threshold=CRY_THRESHOLD)
        if not cry_result["is_cry"]:
            continue  # skip non-cry files
        # CLAP-based scoring
        scores = rater.compute_similarity(wav_path, criteria)
        result_entry = {
            "file": wav_path,
            "class": class_name,
            "cry_confidence": cry_result["cry_confidence"],
            "criteria_scores": {c: s for c, s in scores}
        }
        results.append(result_entry)

# Save results to a JSON file
with open("scores/donateacry_scored.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Processed {len(results)} files. Results saved to scores/donateacry_scored.json.")
