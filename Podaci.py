import numpy as np
import librosa
import os
import pandas as pd
import random
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import HeatKernel
from tqdm import tqdm
from joblib import Parallel, delayed

# OpenMIC-2018: skinuti sa https://zenodo.org/record/1432913
# Raspakirati u folder openmic-2018/ unutar projekta
OPENMIC_DIR = 'openmic-2018'
AUDIO_DIR = os.path.join(OPENMIC_DIR, 'audio')
LABELS_CSV = os.path.join(OPENMIC_DIR, 'openmic-2018-aggregated-labels.csv')

SAMPLES_PER_CLASS = 100
RELEVANCE_THRESHOLD = 0.5
SR_TARGET = 22050
CLIP_SECONDS = 4  # koristimo prvih 4 sekunde svakog 10s klipa

INSTRUMENTS = [
    'accordion', 'banjo', 'bass', 'cello', 'clarinet', 'cymbals', 'drums',
    'flute', 'guitar', 'mallet_percussion', 'mandolin', 'organ', 'piano',
    'saxophone', 'synthesizer', 'trombone', 'trumpet', 'ukulele', 'violin', 'voice'
]


def process_clip(sample_key, label_vec, audio_dir):
    subdir = sample_key[:3]
    filepath = os.path.join(audio_dir, subdir, f"{sample_key}.ogg")
    if not os.path.exists(filepath):
        return None
    try:
        audio, _ = librosa.load(filepath, sr=SR_TARGET, mono=True, duration=CLIP_SECONDS)
        audio = audio.astype(np.float32)
        if len(audio) < 2000:
            return None

        embedder = SingleTakensEmbedding(
            parameters_type="fixed", time_delay=8, dimension=3, stride=100
        )
        vr = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], max_edge_length=5)
        kernel = HeatKernel()

        embedded = embedder.fit_transform(audio).reshape(1, -1, 3)
        diagram = vr.fit_transform(embedded)
        heat = kernel.fit_transform(diagram)[0]

        return heat, label_vec
    except Exception:
        return None


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    print("Ucitavam labele...")
    df = pd.read_csv(LABELS_CSV)

    print("Gradim multi-label matricu...")
    df_pos = df[df['instrument'].isin(INSTRUMENTS)]
    pivot = df_pos.pivot_table(
        index='sample_key',
        columns='instrument',
        values='relevance',
        aggfunc='max',
        fill_value=0.0,
    )
    for inst in INSTRUMENTS:
        if inst not in pivot.columns:
            pivot[inst] = 0.0
    pivot = pivot[INSTRUMENTS]
    label_matrix = (pivot >= RELEVANCE_THRESHOLD).astype(np.float32)

    # Za svaki instrument uzmemo do SAMPLES_PER_CLASS klipova
    selected_keys = set()
    for inst in INSTRUMENTS:
        positive = label_matrix[label_matrix[inst] == 1.0].index.tolist()
        random.shuffle(positive)
        selected_keys.update(positive[:SAMPLES_PER_CLASS])

    selected_keys = list(selected_keys)
    print(f"Ukupno klipova za obradu: {len(selected_keys)}")

    print("Ekstraktujem TDA feature-e (paralelno)...")
    results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(process_clip)(
            key,
            label_matrix.loc[key].values,
            AUDIO_DIR,
        )
        for key in selected_keys
    )

    features, labels = [], []
    for r in results:
        if r is not None:
            feat, lbl = r
            features.append(feat)
            labels.append(lbl)

    features_arr = np.array(features)
    labels_arr = np.array(labels)

    np.save("Heat_diagrams.npy", features_arr)
    np.save("Labele.npy", labels_arr)

    print(f"\nSacuvano {len(features_arr)} uzoraka")
    print(f"Features shape: {features_arr.shape}")
    print(f"Labele shape:   {labels_arr.shape}")
    print("\nDistribucija po klasama:")
    for i, inst in enumerate(INSTRUMENTS):
        count = int(labels_arr[:, i].sum())
        print(f"  {inst:20s}: {count}")
