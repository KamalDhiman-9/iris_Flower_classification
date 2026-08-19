"""
Iris Flower Classification
Classifies iris flowers into Setosa, Versicolor, Virginica using
petal/sepal measurements. Covers EDA, train/test split, model
training, and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
species_names = iris.target_names  # ['setosa' 'versicolor' 'virginica']

df = X.copy()
df["species"] = y.map(dict(enumerate(species_names)))

print("Shape:", df.shape)
print(df.head())
print("\nClass balance:\n", df["species"].value_counts())

# ---------------------------------------------------------
# 2. Explore visually
# ---------------------------------------------------------
sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Iris Feature Relationships", y=1.02)
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.flatten(), X.columns):
    sns.histplot(data=df, x=col, hue="species", kde=True, ax=ax)
plt.tight_layout()
plt.savefig("iris_histograms.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved iris_pairplot.png and iris_histograms.png")

# ---------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 4. Preprocess (scaling helps KNN & Logistic Regression)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. Train multiple classifiers and compare
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

results = {}

for name, model in models.items():
    if name == "Decision Tree":
        # Trees don't need scaling, but it doesn't hurt
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    results[name] = {"model": model, "accuracy": acc, "precision": prec, "preds": preds}

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n",
          classification_report(y_test, preds, target_names=species_names))

# ---------------------------------------------------------
# 6. Visualize confusion matrix for the best model
# ---------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_preds = results[best_name]["preds"]

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=species_names, yticklabels=species_names)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")
print("Saved iris_confusion_matrix.png")
