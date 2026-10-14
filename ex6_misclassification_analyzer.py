"""Exercise 6: Misclassification Confidence Analyzer."""
import numpy as np, pandas as pd
from utils import *

def run():
    d = get_models(); p = d["bpnn"].predict_proba(d["Xte"]); yte = d["yte"]
    pred = (p >= .5).astype(int); bad = np.where(pred != yte)[0]
    if not len(bad): return print("No misclassified samples.")
    df = pd.DataFrame({"Sample": bad + 1, "Actual": [CLASS[a] for a in yte[bad]],
                       "Predicted": [CLASS[a] for a in pred[bad]], "Output": p[bad].round(4),
                       "Confidence": [confidence(v) for v in p[bad]]})
    print(df.to_string(index=False))
    n = (df.Confidence == "High").sum()
    print(f"\nConfidently wrong predictions: {n}" if n else "\nNo confidently wrong predictions.")

if __name__ == "__main__": run()
