import numpy as np
import pandas as pd
from string import printable
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, GRU, Dense, Dropout, Flatten, concatenate
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------
# Config
# ---------------------------------------------------
MAX_LEN = 150
EMBEDDING_DIM = 32
VOCAB_SIZE = len(printable) + 1

# ---------------------------------------------------
# Data preparation (URLs -> sequences)
# ---------------------------------------------------
def urls_to_sequences(url_list):
    url_int_tokens = [[printable.index(ch) + 1 for ch in url if ch in printable] for url in url_list]
    return sequence.pad_sequences(url_int_tokens, maxlen=MAX_LEN)

# Example: Load your CSV
# Must have at least columns ['url', 'result_flag']
train_df = pd.concat([
    pd.read_csv("features/legitimate_train.csv"),
    pd.read_csv("features/phish_train.csv")
]).sample(frac=1).reset_index(drop=True)

test_df = pd.concat([
    pd.read_csv("features/legitimate_test.csv"),
    pd.read_csv("features/phish_test.csv")
]).sample(frac=1).reset_index(drop=True)

X_train = urls_to_sequences(train_df['url'].values)
y_train = train_df['result_flag'].values
X_test = urls_to_sequences(test_df['url'].values)
y_test = test_df['result_flag'].values

# ---------------------------------------------------
# Build CNN + RNN(LSTM) Hybrid Model
# ---------------------------------------------------
input_layer = Input(shape=(MAX_LEN,))

# Shared embedding
embed = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN)(input_layer)

# CNN branch
cnn_branch = Conv1D(filters=64, kernel_size=5, activation='relu')(embed)
cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
cnn_branch = Flatten()(cnn_branch)

# RNN/LSTM branch
rnn_branch = LSTM(64)(embed)

# Combine CNN + RNN
combined = concatenate([cnn_branch, rnn_branch])

# Dense layers for classification
x = Dense(64, activation='relu')(combined)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# Final model
hybrid_model = Model(inputs=input_layer, outputs=output)

# Compile
hybrid_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# ---------------------------------------------------
# Train
# ---------------------------------------------------
history = hybrid_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    verbose=1
)

# ---------------------------------------------------
# Evaluate
# ---------------------------------------------------
loss, acc = hybrid_model.evaluate(X_test, y_test, verbose=0)
print(f"Hybrid CNN+RNN(LSTM) Model Accuracy: {acc:.4f}")

# ---------------------------------------------------
# Save
# ---------------------------------------------------
hybrid_model.save("models/model_CNN_RNN_LSTM_Hybrid.h5")
