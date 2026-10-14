"""Exercise 1: Tumor Classification using BPNN."""
import numpy as np, pandas as pd
from utils import *

def run():
    d = get_models(); bpnn, Xte, yte = d["bpnn"], d["Xte"], d["yte"]
    p = bpnn.predict_proba(Xte); pred = (p >= .5).astype(int)
    print(pd.DataFrame({"Actual": [CLASS[a] for a in yte], "Predicted": [CLASS[a] for a in pred],
                        "Output": p.round(4)}).head(15))
    print(f"\nBPNN Test Accuracy: {acc(bpnn, Xte, yte):.4f}")

if __name__ == "__main__": run()
