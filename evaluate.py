import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# ===== CONFIG =====
DATA_PATH = "features/all_data.csv"   # must have url,label
MODEL_PATHS = {
    "CNN": "models/model_CNN_LSTM.h5",
    "RNN": "models/model_RNN_LSTM.h5",
    "Hybrid": "models/model_CNN_RNN_LSTM_Hybrid.h5"
}
OUTPUT_DIR = "evaluation_results"
MAX_LEN = 150    # adjust based on training

# ===== PREPROCESS =====
def preprocess_urls(urls):
    # simple char-level tokenizer (turn each char into int)
    char_dict = {c: i + 1 for i, c in enumerate(sorted(set("".join(urls))))}
    url_int_tokens = [[char_dict.get(ch, 0) for ch in url] for url in urls]
    return sequence.pad_sequences(url_int_tokens, maxlen=MAX_LEN)

# ===== LOAD DATA =====
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Cannot find data file: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("❌ all_data.csv must have 'url' and 'label' columns.")
    urls = df["url"].astype(str).tolist()
    labels = df["label"].map({"Legitimate": 0, "Phishing": 1}).values
    return urls, labels

# ===== EVALUATE + SAVE =====
def evaluate_and_plot(name, model, X, y_true):
    print(f"\n=== Evaluating {name} ===")
    y_pred_probs = model.predict(X)
    y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=["Legitimate", "Phishing"]
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Create output folder for this model
    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    # Save report
    with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Plot and save confusion matrix
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Legitimate", "Phishing"])
    plt.yticks([0, 1], ["Legitimate", "Phishing"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()

    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{name} ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_dir, "roc_curve.png"))
    plt.close()

    return {"accuracy": acc, "auc": roc_auc, "report": report, "cm": cm}

# ===== MAIN =====
if __name__ == "__main__":
    urls, labels = load_data()
    X = preprocess_urls(urls)

    results_summary = []

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️ Model file for {name} not found at {path}. Skipping...")
            continue
        model = load_model(path)
        metrics = evaluate_and_plot(name, model, X, labels)
        results_summary.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "AUC": metrics["auc"]
        })

    # Save summary
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    print("\n=== Summary ===")
    for row in results_summary:
        print(f"{row['Model']}: Accuracy={row['Accuracy']:.4f}, AUC={row['AUC']:.4f}")
