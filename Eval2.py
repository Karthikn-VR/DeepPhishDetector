import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate_and_plot(model_name, model, X_test, y_test):
    # Predict
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Metrics
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"])
    print(f"\n=== {model_name} Report ===\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create output folder
    out_dir = os.path.join("results", model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Phishing"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix for {model_name} at {save_path}")
