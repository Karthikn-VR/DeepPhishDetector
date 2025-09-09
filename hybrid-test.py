# hybrid_test.py
import numpy as np
from string import printable
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# -------------------------------
# Config
# -------------------------------
MAX_LEN = 150
MODEL_PATH = "models/model_CNN_RNN_LSTM_Hybrid.h5"
VOCAB_SIZE = len(printable) + 1

# -------------------------------
# Utility: URLs → sequences
# -------------------------------
def urls_to_sequences(url_list):
    url_int_tokens = [[printable.index(ch) + 1 for ch in url if ch in printable] for url in url_list]
    return sequence.pad_sequences(url_int_tokens, maxlen=MAX_LEN)

# -------------------------------
# Test URLs (mixed, random order)
# -------------------------------
test_urls = [
    "https://www.google.com",
    "http://paypal-secure-login-update.com",
    "https://github.com",
    "http://secure-verification-paypal-update.net",
    "https://www.openai.com",
    "http://apple-verify-account.com/login",
    "https://www.wikipedia.org",
    "http://microsoftsupport-secure-login.net",
    "https://www.stackoverflow.com",
    "http://bankofamerica-login-update-security.com"
]

# Optional: If you have ground truth (1=Phishing, 0=Legitimate)
# Replace below with actual labels if you want accuracy:
true_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# -------------------------------
# Load model
# -------------------------------
model = load_model(MODEL_PATH)

# -------------------------------
# Predict
# -------------------------------
X_test = urls_to_sequences(test_urls)
pred_probs = model.predict(X_test)
pred_classes = (pred_probs > 0.5).astype("int32").flatten()

# -------------------------------
# Print results
# -------------------------------
print("\nHybrid CNN+RNN+LSTM Model Test Results:\n")
for url, pred in zip(test_urls, pred_classes):
    label = "Phishing" if pred == 1 else "Legitimate"
    print(f"{url} --> {label}")

# -------------------------------
# Accuracy (if ground truth available)
# -------------------------------
if 'true_labels' in locals():
    acc = np.mean(pred_classes == np.array(true_labels))
    print(f"\nTest Accuracy: {acc:.4f}")
