"""Exercise 3: Feature Contribution Analyzer."""
import numpy as np, pandas as pd
from utils import *

def run(idx=0, delta=0.2):
    d = get_models(); mlp, Xte, yte, feats = d["mlp"], d["Xte"], d["yte"], d["feats"]
    x = Xte[idx:idx + 1]; base = mlp.predict_proba(x)[0]; res = []
    for j, f in enumerate(feats):
        xm = x.copy(); xm[0, j] = np.clip(xm[0, j] + delta, 0, 1)   # change ONE feature only
        res.append((f, mlp.predict_proba(xm)[0] - base))
    res.sort(key=lambda r: abs(r[1]), reverse=True)
    print(f"Test sample #{idx+1} | actual={CLASS[yte[idx]]} | base output={base:.4f}\nTop 5 influential features:")
    for f, c in res[:5]: print(f"  {f:28s} change in output = {c:+.4f}")

if __name__ == "__main__": run()
