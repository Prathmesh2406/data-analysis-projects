# 🏦 Bank Loan Default Prediction — KNN & K-Means Analysis

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Academic Project** · Data Analysis & Basic Programming · M.Eng. Sustainable Technology Management  
> SRH University of Applied Sciences, Germany · Semester 1 · 2025

---

## 📋 Overview

This project applies **supervised** and **unsupervised machine learning** to a real-world bank loan dataset of **700 customers** to:

1. **Predict** which customers are likely to default on their loan (KNN Classification)
2. **Segment** customers into risk groups without using labels (K-Means Clustering)

Both approaches are combined to give a complete analytical picture — prediction accuracy from supervised learning, and hidden risk pattern discovery from unsupervised learning.

---

## 🎯 Objectives

| # | Objective | Method |
|---|-----------|--------|
| 1 | Predict loan defaulters from financial features | KNN Classification |
| 2 | Discover natural customer risk segments | K-Means Clustering |
| 3 | Identify which features drive default risk | Correlation Heatmap + EDA |
| 4 | Evaluate and compare both ML approaches | Accuracy, Confusion Matrix, Cluster Crosstab |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Bank loan records (academic dataset) |
| Rows | 700 customers |
| Features | 6 (AGE, EMPLOY, ADDRESS, DEBTINC, CREDDEBT, OTHDEBT) |
| Target | DEFAULTER (1 = defaulted, 0 = not defaulted) |
| Missing values | None |

**Feature descriptions:**

- `AGE` — Customer age (years)
- `EMPLOY` — Years of employment
- `ADDRESS` — Years at current address
- `DEBTINC` — Debt-to-income ratio (%)
- `CREDDEBT` — Credit card debt (thousands)
- `OTHDEBT` — Other debt (thousands)
- `DEFAULTER` — Target: whether the customer defaulted

---

## 🔬 Methodology

```
Raw Data (700 rows)
       │
       ▼
   Data Loading & Exploration
   (df.info, df.describe, value_counts)
       │
       ▼
   Exploratory Data Analysis (EDA)
   ├── Pairplot (feature relationships by class)
   └── Correlation Heatmap (Pearson coefficients)
       │
       ├──────────────────────┐
       ▼                      ▼
 SUPERVISED LEARNING    UNSUPERVISED LEARNING
 KNN Classification      K-Means Clustering
 ├── Train/Test Split     ├── Elbow Method (k=1–10)
 ├── StandardScaler       ├── Fit with k=2
 ├── k=5, Euclidean dist  ├── Centroid Analysis
 └── Accuracy + Report   └── Cluster vs Defaulter Crosstab
       │                      │
       └──────────┬───────────┘
                  ▼
           Combined Insights
```

---

## 📈 Key Results

### Supervised Learning — KNN Classifier

| Metric | Value |
|--------|-------|
| Accuracy | ~XX% *(run notebook to see)* |
| Algorithm | K-Nearest Neighbours (k=5, Euclidean) |
| Train/Test Split | 70% / 30% (stratified) |
| Preprocessing | StandardScaler (required for distance-based KNN) |

### Unsupervised Learning — K-Means Clustering

| Cluster | Risk Profile | Dominant DEBTINC | Dominant CREDDEBT |
|---------|-------------|-----------------|------------------|
| Cluster 0 | Lower Risk | Lower ratio | Lower debt |
| Cluster 1 | Higher Risk | Higher ratio | Higher debt |

> **Conclusion:** KNN is better suited for direct **default prediction** (labelled data).  
> K-Means is valuable for **risk profiling** and customer segmentation without labels.  
> Together they provide both predictive power and interpretable customer groupings.

---

## 🏭 Industry 4.0 & Real-World Relevance

While this project uses a financial dataset, the same ML pipeline maps directly to **smart manufacturing** and **Industry 4.0** use cases:

| This Project | Industry 4.0 Equivalent |
|-------------|------------------------|
| Predict loan defaulters (KNN) | Predict machine failures / defective products |
| Debt-to-income ratio features | Sensor readings / process parameters |
| Customer risk segmentation (K-Means) | Machine health clustering / production line grouping |
| Elbow method for optimal k | Optimising maintenance schedules |
| StandardScaler on financial data | Normalising multi-sensor industrial data |

> This project demonstrates the **transferable ML workflow** from finance to manufacturing analytics — a core competency for **predictive maintenance** and **smart factory** systems.

---

## 🗂️ Repository Structure

```
bank-loan-default-analysis/
│
├── 📓 notebooks/
│   └── bank_loan_analysis.ipynb       # Main analysis notebook (clean version)
│
├── 📁 data/
│   └── BANK_LOAN.csv                  # Dataset (700 rows, 8 columns)
│
├── 📁 outputs/
│   ├── pairplot.png                   # EDA scatter matrix
│   ├── correlation_heatmap.png        # Feature correlation heatmap
│   ├── knn_predicted_classes.png      # KNN decision boundary (2D)
│   ├── elbow_method.png               # K-Means elbow curve
│   ├── cluster_scatter.png            # Cluster visualization with centroids
│   ├── boxplot_debtinc.png            # DEBTINC by cluster
│   └── boxplot_creddebt.png           # CREDDEBT by cluster
│
├── 📄 README.md                       # This file
├── 📄 requirements.txt                # Python dependencies
└── 📄 LICENSE                         # MIT License
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Prathmesh-Singh/bank-loan-default-analysis.git
cd bank-loan-default-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/bank_loan_analysis.ipynb
```

Or open directly in **Google Colab**:  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prathmesh-Singh/bank-loan-default-analysis/blob/main/notebooks/bank_loan_analysis.ipynb)

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
jupyter>=1.0.0
```

---

## 📚 What I Learned

- How to apply the **full supervised ML pipeline**: EDA → preprocessing → model training → evaluation
- Why **feature scaling is critical** for distance-based algorithms like KNN
- The difference between **supervised** (labelled prediction) and **unsupervised** (pattern discovery) learning
- How the **Elbow Method** guides cluster count selection in K-Means
- How to interpret a **Confusion Matrix** and **Classification Report**
- How **domain knowledge** (knowing which features indicate financial risk) improves model interpretation

---

## 🔮 Future Improvements

- [ ] Test additional classifiers: Decision Tree, Random Forest, Logistic Regression
- [ ] Handle class imbalance with SMOTE or class weighting
- [ ] Tune KNN hyperparameter `k` using cross-validation
- [ ] Add interactive visualisations with Plotly
- [ ] Deploy as a simple Streamlit web app

---

## 👤 Author

**Prathmesh Singh**  
M.Eng. Sustainable Technology Management · SRH University of Applied Sciences, Germany  
📫 singhprathmesh2406@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/prathmesh-singh-a9144b245) · [GitHub](https://github.com/Prathmesh-Singh)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, share, and adapt with attribution.
