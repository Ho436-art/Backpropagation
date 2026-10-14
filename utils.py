"""Data loading, normalization, and shared helpers."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from nn import NeuralNet

def load_data(path="data.csv"):
    df = pd.read_csv(path).dropna(axis=1, how="all")          # drop empty trailing column
    y = (df["diagnosis"] == "M").astype(int).values            # Benign=0, Malignant=1
    X = df.drop(columns=["id", "diagnosis"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = MinMaxScaler().fit(Xtr)                               # normalize to [0,1]
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte, list(X.columns)

def confidence(p, t=0.5):
    d = abs(p - t)
    return "Low" if d < 0.15 else "Moderate" if d < 0.35 else "High"

CLASS = {0: "Benign", 1: "Malignant"}
BPNN  = lambda: NeuralNet([30, 16, 1])        # backprop network
MLP   = lambda: NeuralNet([30, 16, 8, 1])     # multilayer (2 hidden layers)
SHLNN = lambda: NeuralNet([30, 8, 1])         # single hidden layer, 8 neurons

def acc(model, X, y):
    return (model.predict(X) == y).mean()

_cache = {}
def get_models():
    """Load data and train BPNN, MLP, SHLNN once (cached)."""
    if not _cache:
        Xtr, Xte, ytr, yte, feats = load_data()
        _cache.update(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, feats=feats,
                      bpnn=BPNN().fit(Xtr, ytr), mlp=MLP().fit(Xtr, ytr), shlnn=SHLNN().fit(Xtr, ytr))
    return _cache
