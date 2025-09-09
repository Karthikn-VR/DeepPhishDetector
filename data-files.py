import pandas as pd
import os

# Paths
FEATURES_DIR = "features"
OUTPUT_FILE = os.path.join(FEATURES_DIR, "all_data.csv")

# Required input CSVs
required_files = [
    "legitimate_train.csv",
    "legitimate_test.csv",
    "phish_train.csv",
    "phish_test.csv"
]

# Verify all files exist
missing = [f for f in required_files if not os.path.exists(os.path.join(FEATURES_DIR, f))]
if missing:
    print("❌ Missing files:", ", ".join(missing))
    print("⚠️ Please ensure all required CSV files are present in the 'features' folder.")
    exit(1)

# Function to load and normalize each CSV
def load_urls(filepath, label_value):
    df = pd.read_csv(filepath)
    # Try to detect the URL column
    url_column_candidates = [col for col in df.columns if 'url' in col.lower()]
    if not url_column_candidates:
        raise ValueError(f"No URL column found in {filepath}")
    url_col = url_column_candidates[0]
    df = df[[url_col]].copy()
    df['label'] = label_value
    df.columns = ['url', 'label']  # Normalize column names
    return df

print("🔹 Loading CSV files...")

legit_train = load_urls(os.path.join(FEATURES_DIR, "legitimate_train.csv"), "Legitimate")
legit_test = load_urls(os.path.join(FEATURES_DIR, "legitimate_test.csv"), "Legitimate")
phish_train = load_urls(os.path.join(FEATURES_DIR, "phish_train.csv"), "Phishing")
phish_test = load_urls(os.path.join(FEATURES_DIR, "phish_test.csv"), "Phishing")

# Combine
print("🔹 Combining data...")
all_data = pd.concat([legit_train, legit_test, phish_train, phish_test], ignore_index=True)

# Shuffle (for randomness)
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined CSV
print(f"💾 Saving combined CSV to {OUTPUT_FILE} ...")
all_data.to_csv(OUTPUT_FILE, index=False)

print("✅ Done! 'features/all_data.csv' now contains only URL and label.")
