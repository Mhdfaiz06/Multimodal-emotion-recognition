import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_RAW = Path("data/raw/archive/TESS Toronto emotional speech set data")
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
LABEL2IDX = {e: i for i, e in enumerate(EMOTIONS)}

def build_manifest():
    records = []
    # Walk through the directory and grab every .wav file
    for wav_path in sorted(DATA_RAW.rglob("*.wav")):
        fname = wav_path.stem
        parts = fname.split("_")
        if len(parts) < 3:
            continue
            
        speaker = parts[0]
        word = "_".join(parts[1:-1])
        emotion = parts[-1].lower()

        if emotion not in LABEL2IDX:
            continue

        records.append({
            "path": str(wav_path),
            "text": word.replace("_", " "),
            "emotion": emotion,
            "label": LABEL2IDX[emotion],
            "speaker": speaker,
        })

    df = pd.DataFrame(records)

    # 70/15/15 Stratified Split for Train/Val/Test
    train, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=42)
    
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    manifest_path = Path("data/processed/manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"Manifest saved to {manifest_path} with {len(df)} samples.")

if __name__ == "__main__":
    build_manifest()