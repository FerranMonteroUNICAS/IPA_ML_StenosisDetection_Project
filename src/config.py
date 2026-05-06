import os

# --- Project Root Path ---
# This automatically finds the base directory of your project relative to this file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
# Based on your file explorer, your images are in 'dataset'
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_IMG_DIR = os.path.join(ROOT_DIR, "dataset")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")

# --- Output Paths ---
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "features.csv")

# --- General Settings ---
RANDOM_STATE = 42