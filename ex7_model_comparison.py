"""Exercise 7: BPNN-MLP-SHLNN Comparative Analysis."""
import numpy as np, pandas as pd
from utils import *

def run():
    d = get_models(); rows = []
    for name, m in (("BPNN", d["bpnn"]), ("MLP", d["mlp"]), ("SHLNN", d["shlnn"])):
        rows.append((name, acc(m, d["Xtr"], d["ytr"]), acc(m, d["Xte"], d["yte"])))
    df = pd.DataFrame(rows, columns=["Model", "TrainAcc", "TestAcc"]).round(4)
    print(df.to_string(index=False)); print(f"\nBest model: {df.sort_values('TestAcc').iloc[-1].Model}")

if __name__ == "__main__": run()
