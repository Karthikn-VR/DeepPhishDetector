import numpy as np
from string import printable
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import os

# -----------------------------
# CONFIG
# -----------------------------
MAX_LEN = 150
MODEL_PATHS = {
    "CNN": "models/model_CNN_LSTM.h5",
    "RNN": "models/model_RNN_LSTM.h5",
    "Hybrid": "models/model_CNN_RNN_LSTM_Hybrid.h5"
}

# -----------------------------
# 1. LOAD MODELS
# -----------------------------
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[name] = load_model(path)
        print(f"{name} model loaded.")
    else:
        print(f"{name} model NOT FOUND at {path}!")

# -----------------------------
# 2. URL TO SEQUENCE
# -----------------------------
def urls_to_sequences(urls):
    tokenized = [[printable.index(ch)+1 for ch in url if ch in printable] for url in urls]
    return sequence.pad_sequences(tokenized, maxlen=MAX_LEN)

# -----------------------------
# 3. PREDICT FUNCTION
# -----------------------------
def predict_urls(urls):
    X = urls_to_sequences(urls)
    results = {}
    for name, model in models.items():
        preds = model.predict(X)
        results[name] = ["Legitimate" if p < 0.5 else "Phishing" for p in preds]
    return results

# -----------------------------
# 4. TEST URLS
# -----------------------------
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

results = predict_urls(test_urls)

# -----------------------------
# 5. PRINT SUMMARY
# -----------------------------
for i, url in enumerate(test_urls):
    print(f"\n🔗 URL: {url}")
    for model_name in models.keys():
        print(f"  {model_name} Prediction: {results[model_name][i]}")
