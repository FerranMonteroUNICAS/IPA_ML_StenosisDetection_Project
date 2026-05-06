# Coronary Artery Stenosis Detection Project

## 1. Project Overview
This project implements an **Image Processing and Analysis (IPA)** pipeline combined with **Machine Learning** to detect and classify stenotic segments in coronary angiographies. The goal is to identify significant arterial narrowing (≥70% stenosis) from grayscale X-ray sequences using traditional computer vision and classical ML models.

## 2. Dataset Description
The dataset consists of angiographic series from **100 patients** with confirmed one-vessel coronary artery disease.
* **Total Images:** 8,325 grayscale frames.
* **Resolution:** Variable (512x512 to 1000x1000 pixels).
* **Annotations:** Ground truth provided via bounding boxes coordinates defining the stenosis in `14_patienID_serie_frame.xml`.
* **Classification:** Binary (Stenosis vs. Non-Stenosis) for each ROI proposal.

## 3. Project Structure
To maintain modularity and allow team collaboration, the project is organized as follows:

```text
Stenosis_Project/
├── data/
│   ├── raw/                # Original patient folders (e.g., 14-001/)
│   └── train_labels.csv    # Expert annotations
├── src/
│   ├── config.py           # Global paths and IPA parameters
│   ├── utils.py            # Image loading and visualization helpers
│   ├── preprocessing.py    # CLAHE, Denoising, and Normalization
│   ├── candidate_gen.py    # Vessel enhancement and ROI extraction
│   └── train_ml.py         # Feature extraction and ML classification
├── output/
│   ├── models/             # Saved .pkl files (Random Forest/XGBoost)
│   └── features.csv        # The master dataset of extracted features
├── requirements.txt        # Python library dependencies
└── README.md               # Project documentation
```
## 4. Pipeline Workflow
1. **Preprocessing:** Enhance contrast using CLAHE and reduce noise to sharpen vessel borders.
2. **ROI Proposal (IPA):** Use Frangi vesselness filters and diameter analysis to find potential narrowing points along the vessel skeleton.
3. **Feature Extraction:** Calculate geometric metrics (Minimum Lumen Diameter, Percent Stenosis) and texture statistics (Entropy, GLCM) for each ROI.
4. **Classification:** A Machine Learning classifier (Random Forest/XGBoost) determines the final status of each candidate ROI.

## 5. Installation & Setup
1. Ensure you have **Python 3.10+** installed.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the environment:
   * **Windows:** `.venv\Scripts\activate`
   * **Mac/Linux:** `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 6. How to Add New Libraries
To keep the team in sync, follow these steps if you need to install a new package:

1. **Install the library** in your local environment:
   ```bash
   pip install [package-name]
   ```
2. **Update the requirements file** to save the new dependency:
   ```bash
   pip freeze > requirements.txt
   ```
3. **Share the updated `requirements.txt`** with the team.
4. **Team Sync:** Colleagues should run the following to update their own environments:
   ```bash
   pip install -r requirements.txt
   ```
   
## 7. Git Workflow & Branch Structure

To ensure smooth collaboration among team members and avoid merge conflicts, this project follows a feature-branch workflow based on our modular pipeline. 

### Branch Tree

```text
Stenosis_Project_Repo/
├── main                 # Stable, fully tested, production-ready pipeline
├── develop              # Main integration branch for all completed features
├── feature/             # Branches for developing new pipeline components
│   ├── preprocessing    # Work related to CLAHE, denoising (preprocessing.py)
│   ├── candidate-gen    # Work on Frangi filters, ROI proposal (candidate_gen.py)
│   ├── feature-extract  # Work on geometric metrics and texture stats
│   ├── classification   # Work on ML models like RF/XGBoost (train_ml.py)
│   └── core-utils       # Setup and updates for config.py and utils.py
├── bugfix/              # Temporary branches for resolving specific issues
└── chore/               # Environment/dependency updates (e.g., requirements.txt)
````

## Team Members
- Teresa Aguilar Gonzàlez
- Mar Gonzàlez Montesinos
- Ferran Montero Sancho