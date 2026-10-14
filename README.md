# Breast Cancer Wisconsin (Diagnostic) – Backpropagation NN Analysis System

Menu-driven Python project using a NumPy backpropagation network (sigmoid activation)
on the Breast Cancer Wisconsin dataset (Benign = 0, Malignant = 1, features min-max normalized).

## Files
| File | Purpose |
|---|---|
| `ex1_tumor_classification.py` | 1. Tumor Classification using BPNN |
| `ex2_prediction_confidence.py` | 2. Prediction Confidence Analyzer |
| `ex3_feature_contribution.py` | 3. Feature Contribution Analyzer |
| `ex4_hidden_neuron_response.py` | 4. Hidden Neuron Response Analyzer |
| `ex5_feature_masking.py` | 5. Input Feature Masking Challenge |
| `ex6_misclassification_analyzer.py` | 6. Misclassification Confidence Analyzer |
| `ex7_model_comparison.py` | 7. BPNN-MLP-SHLNN Comparative Analysis |
| `ex8_menu_exit.py` | 8. Main menu + Exit |
| `nn.py`, `utils.py` | Backprop network; data loading/helpers |
| `data.csv` | Dataset |

## Run
Full menu: `python ex8_menu_exit.py` — or any exercise alone, e.g. `python ex1_tumor_classification.py`

```bash
pip install -r requirements.txt
python ex8_menu_exit.py
```

## Menu
1. Tumor Classification using BPNN  2. Prediction Confidence Analyzer  3. Feature Contribution Analyzer
4. Hidden Neuron Response Analyzer  5. Input Feature Masking Challenge  6. Misclassification Confidence Analyzer
7. BPNN-MLP-SHLNN Comparative Analysis  8. Exit
