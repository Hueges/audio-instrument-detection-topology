import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

INSTRUMENTS = [
    'accordion', 'banjo', 'bass', 'cello', 'clarinet', 'cymbals', 'drums',
    'flute', 'guitar', 'mallet_percussion', 'mandolin', 'organ', 'piano',
    'saxophone', 'synthesizer', 'trombone', 'trumpet', 'ukulele', 'violin', 'voice'
]

np.random.seed(42)

print("Ucitavam OpenMIC VGGish feature-e...")
data = np.load('openmic-2018/openmic-2018.npz', allow_pickle=True)
X_all  = data['X'].mean(axis=1).astype(np.float32)  # (20000, 128)
Y_cont = data['Y_true'].astype(np.float32)           # (20000, 20) float relevance
Y_mask = data['Y_mask']                              # (20000, 20)

# Binarizuj: pozitivno gde je Y_mask=True i relevance >= 0.5
# Gde Y_mask=False, tretiramo kao negativno (nepoznato → 0)
Y_all = ((Y_cont >= 0.5) & Y_mask).astype(np.int32)

# Koristimo sve uzorke koji imaju bar jednu pouzdanu anotaciju
has_annotation = Y_mask.any(axis=1)
X = X_all[has_annotation]
Y = Y_all[has_annotation]

print(f"Dataset: {X.shape}, labele: {Y.shape}")
print("Uzoraka po klasi:")
for i, inst in enumerate(INSTRUMENTS):
    print(f"  {inst:20s}: {int(Y[:, i].sum())}")

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("\nTreniram Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)
clf = MultiOutputClassifier(rf, n_jobs=-1)
clf.fit(X_train, Y_train)

with open("model_vggish.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Sacuvano: model_vggish.pkl")

# Evaluacija
Y_pred = clf.predict(X_val)

print("\n=== Rezultati po klasi (validacioni set) ===")
print(f"{'Instrument':20s}  {'Precision':>9}  {'Recall':>7}  {'F1':>5}  {'Support':>7}")
print("-" * 62)
for i, inst in enumerate(INSTRUMENTS):
    tp = int(((Y_pred[:, i] == 1) & (Y_val[:, i] == 1)).sum())
    fp = int(((Y_pred[:, i] == 1) & (Y_val[:, i] == 0)).sum())
    fn = int(((Y_pred[:, i] == 0) & (Y_val[:, i] == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    sup  = int(Y_val[:, i].sum())
    print(f"  {inst:20s}  {prec:9.3f}  {rec:7.3f}  {f1:5.3f}  {sup:7d}")
