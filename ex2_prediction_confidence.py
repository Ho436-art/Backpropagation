"""Exercise 2: Prediction Confidence Analyzer (MLP)."""
import numpy as np, pandas as pd
from utils import *

def run():
    d = get_models(); mlp, Xte = d["mlp"], d["Xte"]
    p = mlp.predict_proba(Xte); pred = (p >= .5).astype(int)
    df = pd.DataFrame({"Sample": range(1, len(p) + 1), "Output": p.round(4),
                       "Predicted": [CLASS[a] for a in pred], "Confidence": [confidence(v) for v in p]})
    print(df.head(20).to_string(index=False))
    print("\n", df["Confidence"].value_counts().to_string())

if __name__ == "__main__": run()
