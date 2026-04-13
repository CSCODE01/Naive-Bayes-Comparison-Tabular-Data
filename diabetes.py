import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

file_path = "diabetes-data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Outcome']

df = pd.read_csv(file_path, names=columns)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_pos = X_train.copy()
for col in X_train_pos.columns:
    X_train_pos[col] = X_train_pos[col] - X_train_pos[col].min()
X_test_pos = X_test.copy()
for col in X_test_pos.columns:
    X_test_pos[col] = X_test_pos[col] - X_test_pos[col].min()

models = {
    "Gaussian NB": GaussianNB(),
    "Bernoulli NB": BernoulliNB(),
    "Multinomial NB": MultinomialNB()
}

accuracy_scores = []
precision_scores = []

plt.figure(figsize=(15, 4))

for i, (name, model) in enumerate(models.items()):
    if name == "Multinomial NB":
        model.fit(X_train_pos, y_train)
        y_pred = model.predict(X_test_pos)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    accuracy_scores.append(acc)
    precision_scores.append(prec)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(1, 3, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

results_df = pd.DataFrame({
    'Model': models.keys(),
    'Accuracy': accuracy_scores,
    'Precision': precision_scores
}).sort_values(by='Accuracy', ascending=False)

print("\n=== Final Model Ranking ===")
print(results_df.to_string(index=False))