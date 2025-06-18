import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description='Predict immunotherapy response from fungal ITS data.')
parser.add_argument('--count_file', type=str, default='lab_count.csv',
                    help='Path to the fungal ITS count CSV file (default: lab_count.csv)')
parser.add_argument('--metadata_file', type=str, default='lab_metadata.csv',
                    help='Path to the sample metadata CSV file (default: lab_metadata.csv)')
args = parser.parse_args()

# --- Load data ---
lab_count = pd.read_csv(args.count_file, index_col='Unnamed: 0')
lab_metadata = pd.read_csv(args.metadata_file, index_col='Unnamed: 0')

# --- Preprocess relative abundances ---
lab_RA = lab_count.copy()
lab_RA['level'] = lab_RA.index.str[-1]
lab_RA = lab_RA.groupby('level').apply(lambda x: x / x.sum())
lab_RA = lab_RA.fillna(0)
lab_RA = lab_RA.reset_index(level=0, drop=True)

# --- Calculate fungal load ---
fungal_load = lab_count[lab_count.index.str.endswith('p')].sum() + 1

# --- Filter melanoma samples and species-level data ---
samples = lab_metadata[lab_metadata.tumor_type == "melanoma"].index
data = lab_RA.loc[lab_count.index.str.endswith('s'), samples].T
data = data.loc[:, data.astype(bool).sum() >= 2]
data['fungal_load'] = fungal_load[data.index]
data['fungal_load'] = data['fungal_load'] / data['fungal_load'].max()

# --- Define labels (responders vs non-responders) ---
y = lab_metadata.loc[data.index, 'response_r.nr'].map({'r': 1, 'nr': 0})

# --- ML training and cross-validation ---
n_splits = 6
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))

for fold, (train, test) in enumerate(cv.split(data, y)):
    classifier.fit(data.iloc[train], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        data.iloc[test],
        y.iloc[test],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# --- Plot mean ROC ---
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f ± %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# --- Add std dev shading ---
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"± 1 std. dev.",
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Mean ROC curve with variability",
)
ax.legend(loc="lower left", bbox_to_anchor=(1, 0))
plt.tight_layout()
plt.show()
