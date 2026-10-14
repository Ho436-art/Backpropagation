"""Exercise 5: Input Feature Masking Challenge."""
import numpy as np, pandas as pd
from utils import *

def run():
    d = get_models(); bpnn, Xtr, Xte, yte, feats = d["bpnn"], d["Xtr"], d["Xte"], d["yte"], d["feats"]
    base = acc(bpnn, Xte, yte); means = Xtr.mean(0); res = []
    for j, f in enumerate(feats):
        for name, val in (("zero", 0.0), ("mean", means[j])):
            Xm = Xte.copy(); Xm[:, j] = val
            res.append((f, name, base - acc(bpnn, Xm, yte)))
    df = pd.DataFrame(res, columns=["Feature", "Mask", "AccuracyDrop"]).sort_values("AccuracyDrop", ascending=False)
    print(f"Baseline accuracy: {base:.4f}\n", df.head(10).round(4).to_string(index=False))
    r = df.iloc[0]; print(f"\nMax drop: '{r.Feature}' masked with {r.Mask} -> {r.AccuracyDrop:.4f}")

if __name__ == "__main__": run()
