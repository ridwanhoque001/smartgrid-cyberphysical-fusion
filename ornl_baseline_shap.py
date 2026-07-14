"""
================================================================================
Smart Grid Cyber-Physical Attack Dataset
Beginner starter: (1) reproduce a baseline detector, (2) add SHAP explainability
================================================================================

WHAT THIS SCRIPT DOES, IN PLAIN ENGLISH
---------------------------------------
1. Loads the power-system dataset: one subfolder per attack type (benign,
   backdoor, Bruteforce, FDI, ransomware, reverseshell), each containing a
   Physical.csv of PMU-style sensor readings (Freq, Theta, V_A/B/C, I_A/B/C,
   ActivePower, ReactivePower, BreakerStatus).
2. Cleans it (the raw numbers use comma thousands-separators, e.g. "85,732,060",
   which break math until stripped).
3. Trains a Random Forest -- a standard, reliable "first model". This is your
   BASELINE (the result you reproduce).
4. Prints accuracy, F1 score, and a confusion matrix so you can see how well it did.
5. Runs SHAP, which explains WHICH sensor features pushed the model toward
   "attack". This is your NOVEL TWIST.

You do NOT need to understand the model internals to run this. Run it, get
numbers and pictures, then start changing small things.

HOW TO RUN (easiest path = Google Colab, no install pain)
---------------------------------------------------------
1. Go to https://colab.research.google.com  ->  New notebook.
2. In the first cell, install the libraries:
       !pip install pandas scikit-learn shap matplotlib scipy
3. Upload your data folder (keep the per-attack-type subfolder structure).
4. Paste this script into a cell (or upload it) and set DATA_PATH below.
5. Press Run.

WHERE TO GET THE DATA
---------------------
SmartGrid-CyberPhysical-Attack-Dataset (IEEE DataPort, 2025).
Point DATA_PATH at the folder that directly contains the attack-type
subfolders (e.g. "data/Dataset").
================================================================================
"""

import os
import glob
import numpy as np
import pandas as pd

# scikit-learn = the standard machine-learning toolkit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")  # lets us SAVE plots even with no screen (e.g. on a server)
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ SETTINGS
# DATA_PATH must point at the folder that directly contains the attack-type
# subfolders, e.g. "data/Dataset" -> data/Dataset/benign/Physical.csv, etc.
DATA_PATH = "data/Dataset"
LABEL_COLUMN = "attack_type"  # added automatically from each subfolder's name
TASK = "binary"                # "binary" = attack vs not-attack (easiest to start)
OUTPUT_DIR = "outputs"         # where pictures get saved
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------- LOADING DATA
def load_data(path):
    """Load every <attack_type>/Physical.csv under `path`, stacked into one
    table, tagging each row with its attack_type (the subfolder name)."""
    subdirs = sorted(
        d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)
    )
    if not subdirs:
        raise FileNotFoundError(
            f"No attack-type subfolders found inside '{path}'. Expected e.g. "
            f"'{path}/benign/Physical.csv', '{path}/backdoor/Physical.csv', ... "
            "-- check DATA_PATH."
        )

    frames = []
    for d in subdirs:
        attack_type = os.path.basename(d.rstrip("/\\"))
        phys_path = os.path.join(d, "Physical.csv")
        if not os.path.exists(phys_path):
            print(f"  Skipping '{attack_type}': no Physical.csv found.")
            continue
        df = pd.read_csv(phys_path)
        df["attack_type"] = attack_type
        frames.append(df)
        print(f"  Loaded {len(df)} rows from {attack_type}/Physical.csv")

    if not frames:
        raise FileNotFoundError(f"No Physical.csv files found under '{path}'.")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------- CLEANING
def clean_data(df, label_col):
    """Turn messy raw data into clean numbers the model can use."""
    # Some rows contain the text "Infinity" / "inf" -- replace with "missing".
    df = df.replace(["Infinity", "infinity", "inf", "-inf"], np.nan)

    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. Columns are: {list(df.columns)[-5:]}"
            "  <- set LABEL_COLUMN to the right one (often the LAST column)."
        )

    y = df[label_col]                      # the answers (labels)
    X = df.drop(columns=[label_col])       # the sensor readings (features)

    # @timestamp is a raw clock value, not a sensor reading -- drop it so the
    # model learns from physics, not from when each file happened to be recorded.
    X = X.drop(columns=["@timestamp"], errors="ignore")

    # These readings use comma thousands-separators (e.g. "85,732,060"), which
    # pd.to_numeric would otherwise treat as garbage and turn into NaN.
    X = X.apply(
        lambda col: col.str.replace(",", "", regex=False) if col.dtype == object else col
    )

    # Force every feature column to be a number; anything weird becomes NaN.
    X = X.apply(pd.to_numeric, errors="coerce")

    # IMPORTANT: pandas may have already parsed "Infinity" text as real infinity.
    # Models reject infinity, so turn any inf into "missing" too.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely empty, then fill remaining gaps with 0.
    X = X.dropna(axis=1, how="all").fillna(0)

    return X, y


def make_binary(y):
    """Collapse labels into 'attack' (1) vs 'not attack' (0)."""
    NOT_ATTACK = {"benign", "natural", "no events", "noevents", "0"}

    def is_attack(v):
        s = str(v).strip().lower()
        # Any attack_type folder name (backdoor, Bruteforce, FDI, ransomware,
        # reverseshell, ...) is an attack; only "benign" (or ORNL's
        # natural/no-events labels) counts as not-attack.
        return 0 if s in NOT_ATTACK else 1
    return y.apply(is_attack)


# ------------------------------------------------------------------- MAIN
def main():
    print("\n=== STEP 1: load data ===")
    df = load_data(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    print("\n=== STEP 2: clean data ===")
    X, y_raw = clean_data(df, LABEL_COLUMN)

    if TASK == "binary":
        y = make_binary(y_raw)
        print("Label counts (0 = not attack, 1 = attack):")
    else:
        y = LabelEncoder().fit_transform(y_raw.astype(str))
        print("Label counts (multiclass):")
    print(pd.Series(y).value_counts())

    print("\n=== STEP 3: split into train / test ===")
    # 80% to learn from, 20% held back to test honestly.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    print("\n=== STEP 4: train the BASELINE model (Random Forest) ===")
    # class_weight="balanced" tells the model to stop ignoring the rare class.
    # Without it, attacks outnumber benign ~5:1 and the model under-detects benign.
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    print("Done training.")

    print("\n=== STEP 5: evaluate the baseline ===")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    f1_w = f1_score(y_test, preds, average="weighted")
    f1_m = f1_score(y_test, preds, average="macro")
    print(f"Weighted F1: {f1_w:.4f}   (flattered by class imbalance)")
    print(f"Macro F1:    {f1_m:.4f}   <-- the HONEST number (treats both classes equally)")

    # Save a confusion matrix picture.
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix (baseline)")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    print(f"Saved {OUTPUT_DIR}/confusion_matrix.png")

    print("\n=== STEP 6: the NOVEL TWIST -- explain WHY (SHAP) ===")
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run:  pip install shap   then re-run.")
        return

    # Explaining every row is slow; a sample of the test set is plenty.
    sample = X_test.sample(min(500, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Pull out the "attack" class explanations. SHAP returns different shapes
    # depending on its version, so we handle both:
    if isinstance(shap_values, list):
        sv = shap_values[1]                  # older SHAP: list, [class0, class1]
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:                     # newer SHAP: (rows, features, classes)
            sv = sv[:, :, 1]                 # keep the attack class

    # This bar chart ranks the sensor features that most drive "attack" calls.
    plt.figure()
    shap.summary_plot(sv, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR}/shap_feature_importance.png  <-- your novelty figure!")

    print("""
WHAT TO DO NEXT (your research story):
  - Look at the SHAP chart: which PMU measurements top the list?
  - Re-run with TASK = "multiclass" and make one SHAP chart PER attack type.
  - Compare the charts: do different attacks have different 'fingerprints'?
  - Ask: do these features make physical sense for that attack? (talk to your PI)
That comparison IS your contribution.
""")


if __name__ == "__main__":
    main()
