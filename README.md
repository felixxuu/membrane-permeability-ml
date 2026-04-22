# Membrane Permeability Prediction — BBB Baseline

A machine learning baseline for predicting blood-brain barrier (BBB) permeability from molecular structure (SMILES).

## Motivation

Membrane permeability is a key determinant in drug discovery — especially for central nervous system (CNS) targets, where drugs must cross the blood-brain barrier to be effective. This project establishes a reproducible baseline for BBB permeability classification using classical cheminformatics descriptors and a Random Forest model.

More broadly, this work is an entry point into a longer research interest: building structured, ML-ready datasets that capture the physical-chemical determinants of drug-membrane interactions.

## Dataset

**B3DB** (Blood-Brain Barrier Database) — 7,807 molecules with binary BBB+/BBB- labels, curated from published BBB permeability studies. Source: [theochem/B3DB](https://github.com/theochem/B3DB).

- **Final usable size after validity check**: 7,805 molecules (2 filtered by RDKit sanitization)
- **Class distribution**: BBB+ (4,956) vs BBB- (2,849) — mild imbalance (~1.74:1)
- **Continuous logBB values** are only available for 1,058 molecules; the binary classification task is prioritized for coverage.

## Methods

### Featurization
- **Morgan fingerprints** (ECFP4 equivalent) with radius=2, 2048 bits, computed via RDKit's `MorganGenerator`.

### Model
- **Random Forest Classifier** with 200 trees, `class_weight='balanced'` to handle class imbalance, stratified 80/20 train-test split.

### Evaluation
- Accuracy, ROC-AUC, per-class precision/recall, confusion matrix.

## Results

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.8834 |
| ROC-AUC    | 0.9612 |

![ROC Curve](results/roc_curve.png)

![Confusion Matrix](results/confusion_matrix.png)

### Per-class performance

|        | Precision | Recall | F1    |
|--------|-----------|--------|-------|
| BBB-   | 0.89      | 0.78   | 0.83  |
| BBB+   | 0.88      | 0.94   | 0.91  |

The model shows **asymmetric recall**: high sensitivity for BBB+ (0.94) but lower sensitivity for BBB- (0.78), indicating a slight bias toward predicting permeability. For drug discovery, this means higher false-positive risk when screening out non-CNS candidates.

## Error Analysis

The most confident misclassifications cluster around two molecular archetypes:

1. **Complex natural products** with dense stereochemistry (macrolides, terpenoids).
2. **Antibiotic-class compounds** (e.g., dextramycin, plicamycin) with MW > 500 Da.

These molecules exceed the typical small-molecule drug space and are underrepresented in the training distribution. Morgan fingerprints at radius=2 cannot fully capture the long-range 3D interactions governing permeability in such scaffolds.

## Limitations and Future Directions

- **Featurization ceiling**: Morgan fingerprints ignore 3D conformation and stereochemistry at scale. Graph neural networks (e.g., Chemprop, MPNN) may close this gap.
- **Data distribution bias**: The B3DB corpus skews toward classical drug-like chemistry. Generalization to peptides, large natural products, and biologics requires expanded training data.
- **Binary label reduction**: Collapsing logBB to BBB+/BBB- discards gradient information. A future regression track on the 1,058 logBB-labeled molecules would complement classification.
- **Mechanism blind spot**: This model treats permeability as a property of the molecule alone. In reality, BBB permeability depends on lipid composition, transporter expression, and tissue state — none of which are represented here. Pairing chemical features with membrane-composition data is a promising future direction.

## Reproducibility

```bash
conda create -n memperm python=3.11 -y
conda activate memperm
pip install rdkit pandas scikit-learn xgboost matplotlib seaborn jupyter joblib
```

Dataset: download `B3DB_classification.tsv` from [theochem/B3DB](https://github.com/theochem/B3DB/blob/main/B3DB/B3DB_classification.tsv) into `data/`.

Run notebooks in order:
- `notebooks/01_explore_data.ipynb` — data loading and exploration
- `notebooks/02_baseline_model.ipynb` — featurization, training, evaluation

## Project Structure
membrane-permeability-ml/
├── README.md
├── data/
│   └── B3DB_classification.tsv
├── notebooks/
│   ├── 01_explore_data.ipynb
│   └── 02_baseline_model.ipynb
├── results/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── rf_baseline_model.pkl
└── src/
---

*Author*: Ying (Felix) Xu  
*Context*: Initial exploration toward building structured, ML-ready datasets for drug-membrane interaction prediction.