import os

# --- Project Root Path ---
# This automatically finds the base directory of your project relative to this file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
# Based on your file explorer, your images are in 'dataset'
RAW_IMG_DIR = os.path.join(ROOT_DIR, "organized_dataset")

# --- General Settings ---
RANDOM_STATE = 42