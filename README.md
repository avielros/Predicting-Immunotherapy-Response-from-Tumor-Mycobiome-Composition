# Predicting-Immunotherapy-Response-from-Tumor-Mycobiome-Composition

## 🧪 Project Summary

This project explores whether the composition of the fungal microbiome (mycobiome) in melanoma tumors can predict patient response to immunotherapy. The analysis is based on real ITS sequencing data, comparing tumor samples from responders (r) and non-responders (nr) to immune checkpoint blockade therapy.

We use machine learning (Random Forest classifier) to test whether fungal species abundance and overall fungal load are associated with treatment outcomes. The goal is to provide a reproducible tool that can support further research into the tumor microenvironment and host–microbe interactions in cancer.

---

## 📚 Research Background

Fungi, as part of the tumor microbiome, are emerging as potential influencers of cancer progression and therapy response. Although bacterial contributions have been more studied, the role of fungi remains under-characterized.

This project builds on recent evidence that fungal communities vary across cancer types and may modulate immune responses. We focus on melanoma due to the availability of ITS data and its frequent treatment with immunotherapy.

### 🔗 References

- Narunsky-Haziza et al. (2022). *Pan-cancer analyses reveal cancer-type-specific fungal ecologies and bacteriome interactions*. Cell.  
- Nejman et al. (2020). *The human tumor microbiome is composed of tumor type–specific intracellular bacteria*. Science.

---

## 💡 What Does This Project Do?

This project implements a pipeline to:

- Load ITS sequencing results and sample metadata.
- Normalize fungal species abundances (relative abundance per sample).
- Integrate fungal load data into the analysis.
- Train and cross-validate a Random Forest classifier to predict immunotherapy response.
- Visualize ROC curves and calculate mean AUC with standard deviation.

This tool will help:

- Researchers investigating fungal roles in the tumor microenvironment.
- Labs wishing to integrate mycobiome analysis into cancer immunology workflows.
- Extendable pipelines for other tumor types or omics data.

---

## 📥 Input and 📤 Output

### Input Files

- `lab_count.csv`: Raw ITS counts for fungal taxa in each tumor sample.
- `lab_metadata.csv`: Metadata including tumor type (`tumor_type`) and immunotherapy response (`response_r.nr`).

> Samples are indexed with a suffix: `'s'` = species-level, `'p'` = phylum-level.

### Output

- ROC curves per fold and mean ROC with AUC ± standard deviation.
- Feature matrix after filtering and transformation.
- (Optional) Feature importance plots from the trained classifier.

---

## ⚙️ How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tumor-mycobiome-immunotherapy-response.git
cd tumor-mycobiome-immunotherapy-response
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Requirements.txt:
```txt
pandas
numpy
seaborn
matplotlib
scikit-learn
```

### 3.Run the Main Script
Ensure both lab_count.csv and lab_metadata.csv are in the root directory.
```bash
python ML_ICI_Response.py
```
---
### Repository Structure
```bash

.
├── lab_count.csv                # ITS fungal counts per sample
├── lab_metadata.csv            # Sample metadata
├── ML_ICI_Response.py            # Core ML pipeline
├── requirements.txt
└── README.md
```
## 🎯 Project Goals and Extensions

- Demonstrate the feasibility of predicting therapy response from fungal composition.
- Provide a reusable ML pipeline for lab researchers.

---


## 🧩 Course Info

This project was developed as part of the course **[WIS Python programming course started in 2025.03]**.  

🔗 [Link to the course repository](https://github.com/Code-Maven/wis-python-course-2025-03)
