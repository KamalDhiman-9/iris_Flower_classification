# 🌸 Iris Flower Classification

A classic machine learning project that classifies iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on petal and sepal measurements.

## 📌 Objective

Build and compare simple classification models to predict iris species from four numeric features, and evaluate them using standard classification metrics.

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) / `scikit-learn` built-in dataset
- **Samples:** 150 (50 per species)
- **Features:**
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target:** Species (Setosa / Versicolor / Virginica)

## 🛠️ Approach

1. **Load & explore** — inspect the dataset structure and class balance
2. **Visualize** — pairplots and histograms to understand feature separability across species
3. **Split** — 80/20 train/test split (stratified to preserve class balance)
4. **Preprocess** — feature scaling with `StandardScaler` (for distance/gradient-based models)
5. **Train** — three classifiers trained and compared:
   - Logistic Regression
   - K-Nearest Neighbors (K=5)
   - Decision Tree
6. **Evaluate** — accuracy, macro precision, confusion matrix, and full classification report for each model

## 📈 Results

| Model | Accuracy | Precision (macro) |
|---|---|---|
| Logistic Regression | 93.3% | 0.933 |
| K-Nearest Neighbors | 93.3% | 0.944 |
| Decision Tree | 93.3% | 0.933 |

All three models perform comparably well. **Setosa is perfectly separable** in every model (linearly distinct from the other two species), while most misclassifications occur between **Versicolor and Virginica**, which overlap in petal dimensions.

### Visuals

- `iris_pairplot.png` — pairwise feature relationships by species
- `iris_histograms.png` — feature distributions by species
- `iris_confusion_matrix.png` — confusion matrix for the best-performing model

## 🚀 Usage

```bash
pip install scikit-learn pandas matplotlib seaborn
python iris_classification.py
```

The script prints dataset stats, trains all three models, prints evaluation metrics for each, and saves the visualizations as PNG files in the working directory.

## 🧠 Skills Gained

- Exploratory data analysis on numeric data
- Train/test splitting and feature scaling
- Training and comparing multiple classification algorithms
- Model evaluation with accuracy, precision, and confusion matrices

## 📁 Project Structure

```
├── iris_classification.py       # Main script
├── iris_pairplot.png            # EDA: pairwise feature plots
├── iris_histograms.png          # EDA: feature distributions
├── iris_confusion_matrix.png    # Evaluation: confusion matrix
└── README.md
```

---
*Part of my ML learning journey — #60DayClaudeChallenge*
