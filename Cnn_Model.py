import pandas as pd
import numpy as np
from string import printable
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# -----------------------------
# CONFIG
# -----------------------------
MAX_LEN = 150
VOCAB_SIZE = len(printable) + 1
EMBED_DIM = 64
EPOCHS = 20
BATCH_SIZE = 64
MODEL_PATH = "models/model_CNN_LSTM.h5"

os.makedirs("models", exist_ok=True)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
legit_train = pd.read_csv("features/legitimate_train.csv")
legit_test = pd.read_csv("features/legitimate_test.csv")
phish_train = pd.read_csv("features/phish_train.csv")
phish_test = pd.read_csv("features/phish_test.csv")

train_urls = pd.concat([legit_train.iloc[:,0], phish_train.iloc[:,0]], axis=0)
test_urls = pd.concat([legit_test.iloc[:,0], phish_test.iloc[:,0]], axis=0)

y_train = np.concatenate([np.zeros(len(legit_train)), np.ones(len(phish_train))])
y_test = np.concatenate([np.zeros(len(legit_test)), np.ones(len(phish_test))])

# -----------------------------
# 2. URL TO SEQUENCE
# -----------------------------
def urls_to_sequences(urls):
    tokenized = [[printable.index(ch)+1 for ch in url if ch in printable] for url in urls]
    return sequence.pad_sequences(tokenized, maxlen=MAX_LEN)

X_train = urls_to_sequences(train_urls)
X_test = urls_to_sequences(test_urls)

# -----------------------------
# 3. BUILD CNN + LSTM MODEL
# -----------------------------
input_layer = Input(shape=(MAX_LEN,))
x = Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN)(input_layer)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = LSTM(64)(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# -----------------------------
# 4. TRAIN
# -----------------------------
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
mc = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, mc],
    verbose=1
)

# -----------------------------
# 5. EVALUATE
# -----------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# -----------------------------
# 6. PREDICT FUNCTION
# -----------------------------
def predict_urls(urls):
    X = urls_to_sequences(urls)
    preds = model.predict(X)
    return ["Legitimate" if p < 0.5 else "Phishing" for p in preds]

# -----------------------------
# 7. TEST 10 RANDOM URLS
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
for url, r in zip(test_urls, results):
    print(f"{url} --> {r}")
