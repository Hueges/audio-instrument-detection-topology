"""
Klasifikuj instrumente u audio fajlu koristeći VGGish embeddinge.

Pipeline:
  1. Librosa ucitava audio → resample na 16kHz (VGGish standard)
  2. torchvggish ekstraktuje 128-dim embeddinge po 0.96s segmentima
  3. Srednja vrednost svih segmenata → jedan vektor (1, 128)
  4. Random Forest predviđa verovatnoce za 20 instrumenata
  5. Ispis svih instrumenata iznad praga pouzdanosti

Upotreba:
  python Klasifikuj.py <audio_fajl>
"""

import sys
import os
import pickle
import numpy as np
import librosa
import torch
from torchvggish import vggish, vggish_input

INSTRUMENTS = [
    'accordion', 'banjo', 'bass', 'cello', 'clarinet', 'cymbals', 'drums',
    'flute', 'guitar', 'mallet_percussion', 'mandolin', 'organ', 'piano',
    'saxophone', 'synthesizer', 'trombone', 'trumpet', 'ukulele', 'violin', 'voice'
]

THRESHOLD = 0.35
SR_VGGISH = 16000  # VGGish ocekuje 16kHz

_vggish_model = None


def get_vggish():
    global _vggish_model
    if _vggish_model is None:
        print("  Ucitavam VGGish model...")
        _vggish_model = vggish()
        _vggish_model.eval()
    return _vggish_model


def extract_features(filepath):
    """Vraca (1, 128) VGGish embedding srednje vrednosti svih segmenata."""
    audio, _ = librosa.load(filepath, sr=SR_VGGISH, mono=True)
    audio = audio.astype(np.float32)

    examples = vggish_input.waveform_to_examples(audio, SR_VGGISH)
    if len(examples) == 0:
        return None

    model = get_vggish()
    with torch.no_grad():
        embeddings = model(torch.FloatTensor(examples))

    return embeddings.mean(0).numpy().reshape(1, -1)  # (1, 128)


def klasifikuj(filepath):
    if not os.path.exists(filepath):
        print(f"Greska: fajl '{filepath}' ne postoji.")
        return {}

    if not os.path.exists("model_vggish.pkl"):
        print("Greska: model_vggish.pkl ne postoji. Pokreni prvo Train.py.")
        return {}

    print(f"\nAnaliziram: {filepath}")

    with open("model_vggish.pkl", "rb") as f:
        clf = pickle.load(f)

    print("Ekstraktujem VGGish feature-e...")
    feat = extract_features(filepath)
    if feat is None:
        print("Greska: nije moguce ekstraktovati feature-e.")
        return {}

    probs = np.array([
        est.predict_proba(feat)[0][1]
        for est in clf.estimators_
    ])

    return {
        INSTRUMENTS[i]: float(probs[i])
        for i in range(len(INSTRUMENTS))
        if probs[i] >= THRESHOLD
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Upotreba: python Klasifikuj.py <audio_fajl>")
        sys.exit(1)

    rezultati = klasifikuj(sys.argv[1])

    print("\n" + "=" * 45)
    print("       DETEKTOVANI INSTRUMENTI")
    print("=" * 45)
    if not rezultati:
        print("Nisu detektovani poznati instrumenti.")
    else:
        for naziv, pouzdanost in sorted(rezultati.items(), key=lambda x: -x[1]):
            bar = '█' * int(pouzdanost * 30)
            print(f"  {naziv:20s} {bar:30s} {pouzdanost:.1%}")
    print("=" * 45)
