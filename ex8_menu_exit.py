"""Exercise 8: Main menu (runs exercises 1-7) with Exit option."""
import ex1_tumor_classification as e1, ex2_prediction_confidence as e2, ex3_feature_contribution as e3
import ex4_hidden_neuron_response as e4, ex5_feature_masking as e5
import ex6_misclassification_analyzer as e6, ex7_model_comparison as e7

MENU = {"1": ("Tumor Classification using BPNN", e1), "2": ("Prediction Confidence Analyzer", e2),
        "3": ("Feature Contribution Analyzer", e3), "4": ("Hidden Neuron Response Analyzer", e4),
        "5": ("Input Feature Masking Challenge", e5), "6": ("Misclassification Confidence Analyzer", e6),
        "7": ("BPNN-MLP-SHLNN Comparative Analysis", e7)}

while True:
    print("\n=== Breast Cancer BPNN System ===")
    for k, (n, _) in MENU.items(): print(f"{k}. {n}")
    print("8. Exit")
    c = input("Choice: ").strip()
    if c == "8": print("Goodbye!"); break
    if c in MENU: print(f"\n--- {MENU[c][0]} ---"); MENU[c][1].run()
    else: print("Invalid choice.")
