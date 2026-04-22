# Membrane Permeability Prediction — From Statistical Baselines Toward Physics-Informed Modeling

A comparative study of molecular permeability prediction across the blood-brain barrier, exploring the interface between statistical machine learning and physics-based modeling of membrane transport.

## Scientific Context

Membrane permeability is governed by well-understood physical processes — lipid-water partitioning, molecular diffusion, hydrogen-bond desolvation, cavity formation — that admit both mechanistic description (Abraham LFER, solution-diffusion theory, Stokes-Einstein) and data-driven prediction (molecular fingerprints, graph neural networks).

Most modern cheminformatics pipelines treat permeability as a purely statistical prediction problem. However, the field's foundational work (Clark, Abraham, van de Waterbeemd) demonstrated that a small set of physicochemical descriptors — logP, TPSA, hydrogen-bonding capacity — captures much of permeability variance through linear free energy relationships (LFER).

This project revisits that tension. On a standardized BBB dataset, it compares:

1. A statistical baseline (Morgan fingerprints + Random Forest)
2. A physics-based linear model (LFER-style on Abraham-like descriptors) *(in progress)*
3. A hybrid model combining physics priors with ML residual correction *(in progress)*

The goal is not a state-of-the-art score, but a transparent comparison of what each paradigm captures — and where physics fails in ways that ML can complement.

## Dataset

**B3DB** (Blood-Brain Barrier Database) — 7,807 curated molecules with binary BBB+/BBB- labels and 1,058 with continuous logBB values.
Source: [theochem/B3DB](https://github.com/theochem/B3DB).

- After RDKit sanitization: **7,805 molecules**
- Class distribution: BBB+ (4,956) / BBB- (2,849), mild imbalance (~1.74:1)
- Continuous logBB subset enables parallel regression analysis

## Current Results — Statistical Baseline

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.8834 |
| ROC-AUC    | 0.9612 |

<p align="center">
  <img src="results/roc_curve.png" alt="ROC Curve" width="500"/>
</p>

<p align="center">
  <img src="results/confusion_matrix.png" alt="Confusion Matrix" width="450"/>
</p>

### Per-class performance

|        | Precision | Recall | F1    |
|--------|-----------|--------|-------|
| BBB-   | 0.89      | 0.78   | 0.83  |
| BBB+   | 0.88      | 0.94   | 0.91  |

**Observation**: The model shows asymmetric recall — higher sensitivity for BBB+ (0.94) than for BBB- (0.78). For downstream drug-discovery applications, this translates to elevated false-positive risk when screening non-CNS candidates.

## Error Analysis — When Fingerprint Models Fail

The most confident misclassifications cluster into two archetypes:

1. **Complex natural products** with stereochemically dense scaffolds (macrolides, terpenoids)
2. **Antibiotic-class compounds** (dextramycin, plicamycin) with MW > 500 Da

These molecules exceed typical small-molecule drug space and are underrepresented in the training distribution. Morgan fingerprints at radius=2 cannot capture long-range stereochemical interactions that govern permeability in such scaffolds.

This failure mode motivates the **physics-informed direction** below.

## Roadmap — Physics-Informed Extension (In Progress)

Planned notebook `03_physics_informed_model.ipynb` will add:

- **LFER baseline**: A linear regression on Abraham-like physicochemical descriptors (logP, TPSA, MW, HBD, HBA, rotatable bonds, molar refractivity). Each coefficient carries a physical interpretation that can be compared against the published literature.
- **Residual analysis**: Systematic examination of where the physics-only model fails, stratified by molecular class.
- **Hybrid mechanistic-statistical model**: LFER prediction + ML residual correction, quantifying how much permeability variance requires empirical rather than mechanistic description.

The aim is a transparent decomposition: *how much of membrane permeability is physics, and how much is empirical correction?*

## Project Structure

```
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
```


## Reproducibility

```bash
conda create -n memperm python=3.11 -y
conda activate memperm
pip install rdkit pandas scikit-learn xgboost matplotlib seaborn jupyter joblib
```

Download `B3DB_classification.tsv` from [theochem/B3DB](https://github.com/theochem/B3DB/blob/main/B3DB/B3DB_classification.tsv) into `data/`.

Run notebooks in order:
- `notebooks/01_explore_data.ipynb` — data exploration and quality control
- `notebooks/02_baseline_model.ipynb` — featurization, training, evaluation

## Limitations and Future Directions

- **Featurization ceiling**: Morgan fingerprints ignore 3D conformation and stereochemistry at scale. Graph neural networks (e.g., Chemprop, MPNN) may close this gap.
- **Training distribution bias**: B3DB skews toward classical drug-like chemistry. Generalization to peptides, natural products, and biologics requires expanded training data.
- **Mechanism blind spot**: Current models treat permeability as a property of the molecule alone. In reality, permeability depends on lipid composition, transporter expression, and membrane state — none of which are represented here. Pairing molecular descriptors with explicit membrane biophysics is a promising future direction.

---

*Author*: Ying (Felix) Xu  
*Context*: Exploration at the intersection of membrane biophysics, physical chemistry, and machine learning — toward structured, physics-informed datasets for molecular-membrane interaction modeling.