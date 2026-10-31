import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)

# 1. LOAD THE DATA: Breast Cancer Wisconsin (built into scikit-learn)
#    569 patients, 30 numeric measurements of a tumour,
#    target: 0 = malignant (cancer), 1 = benign (not cancer)
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print("Rows:", X.shape[0], "| Features:", X.shape[1])
print("Malignant:", int((y == 0).sum()), "| Benign:", int((y == 1).sum()))

# Save a copy of the dataset in the data folder
pd.concat([X, y.rename("target")], axis=1).to_csv("data/breast_cancer.csv", index=False)

# 2. SPLIT: 80% to learn from, 20% to test on (never seen before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining rows:", len(X_train), "| Testing rows:", len(X_test))

# 3. SCALE: fit the scaler on TRAINING data only, then apply to both
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. BUILD AND TRAIN THE ANN
#    Input: 30 neurons, Hidden 1: 16 (ReLU), Hidden 2: 8 (ReLU), Output: 1
ann = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                    solver="adam", max_iter=500, random_state=42)
ann.fit(X_train_s, y_train)
print("\nTraining finished after", ann.n_iter_, "iterations")

# 5. EVALUATE on unseen test data
pred = ann.predict(X_test_s)
print("\n" + "=" * 55)
print("RESULTS ON UNSEEN TEST DATA")
print("=" * 55)
print("Train accuracy:", round(accuracy_score(y_train, ann.predict(X_train_s)) * 100, 2), "%")
print("Test accuracy :", round(accuracy_score(y_test, pred) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=data.target_names))

# 6. PICTURES: loss curve and confusion matrix
plt.figure(figsize=(7, 4))
plt.plot(ann.loss_curve_)
plt.xlabel("Iteration (epoch)")
plt.ylabel("Loss (error)")
plt.title("ANN training loss")
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()

ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=data.target_names, cmap="Blues"
)
plt.title("ANN confusion matrix (test data)")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved loss_curve.png and confusion_matrix.png")

# 7. EXPERIMENT: does the size of the network matter?
print("\n" + "=" * 55)
print("EXPERIMENT: different hidden layer sizes")
print("=" * 55)
rows = []
for layers in [(4,), (8,), (16, 8), (32, 16)]:
    m = MLPClassifier(hidden_layer_sizes=layers, max_iter=500, random_state=42)
    m.fit(X_train_s, y_train)
    rows.append([str(layers),
                 round(accuracy_score(y_train, m.predict(X_train_s)) * 100, 2),
                 round(accuracy_score(y_test, m.predict(X_test_s)) * 100, 2)])
print(pd.DataFrame(rows, columns=["Hidden layers", "Train %", "Test %"]).to_string(index=False))
