import os

# --- Project Root Path ---
# This automatically finds the base directory of your project relative to this file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
# Based on your file explorer, your images are in 'dataset'
RAW_IMG_DIR = os.path.join(ROOT_DIR, "dataset_subtracted")

CLAHE_IMG_DIR = os.path.join(ROOT_DIR, "preprocessing/clahe")

NLMEANS_IMG_DIR = os.path.join(ROOT_DIR, "preprocessing/nlmeans")

NLMEANS_IMG_DIR_2 = os.path.join(ROOT_DIR, "preprocessing_with_denoising/nlmeans")
# --- General Settings ---
RANDOM_STATE = 42