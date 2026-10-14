"""Exercise 4: Hidden Neuron Response Analyzer (SHLNN)."""
import numpy as np, pandas as pd
from utils import *

def run():
    d = get_models(); H, yte = d["shlnn"].hidden(d["Xte"]), d["yte"]
    print("Hidden neuron activation matrix (first 10 samples):")
    print(pd.DataFrame(H[:10], columns=[f"N{i+1}" for i in range(H.shape[1])]).round(3))
    print(f"\nBenign    -> neuron N{H[yte == 0].mean(0).argmax()+1} (avg {H[yte == 0].mean(0).max():.3f})")
    print(f"Malignant -> neuron N{H[yte == 1].mean(0).argmax()+1} (avg {H[yte == 1].mean(0).max():.3f})")

if __name__ == "__main__": run()
