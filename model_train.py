# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# %%
df = pd.read_csv(r"C:\Users\anil kumar singh\OneDrive\Desktop\Vendor Performance\creditcard_cleaned.csv")

print(df.head())
print(df.shape)

# %%
print(df["Class"].value_counts())

# %%
X = df.drop("Class", axis=1)

y = df["Class"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# %%
cm = confusion_matrix(y_test, y_pred)

print(cm)

# %%
print(classification_report(y_test, y_pred))

# %%
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# %%
sample = X_test.iloc[[0]]

prediction = model.predict(sample)

print(prediction)

# %%
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print(coefficients.sort_values(
    by="Coefficient",
    ascending=False
))

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.show()

# %%
model.fit(X_train, y_train)

# %%
new_transaction = pd.read_csv(r"C:\Users\anil kumar singh\Downloads\new_credit_card.csv")

# %%
new_prediction = model.predict("C:\\Users\\anil kumar singh\\Downloads\\New__credit_card.csv")
print(new_prediction)

# %%
new_transaction = pd.DataFrame({
    'Time': [50000],
    'V1': [-2.3],
    'V2': [1.5],
    'V3': [-1.2],
    'V4': [0.5],
    'V5': [-0.8],
    "V6": [0.3],
    'V7': [-0.4],
    'V8': [0.1],
    'V9': [-0.2],
    'V10': [0.6],
    'V11': [-0.1],
    'V12': [0.4],
    'V13': [-0.3],
    'V14': [0.2],
    'V15': [-0.5],
    'V16': [0.7],
    'V17': [-0.6],
    'V18': [0.3],
    'V19': [-0.4],
    'V20': [0.2],
    'V21': [-0.1],
    'V22': [0.5],
    'V23': [-0.2],
    'V24': [0.1],
    'V25': [-0.3],
    'V26': [0.4],
    'V27': [-0.2], 
    'V28': [0.02],
    'Amount': [2500]
})

# %%
prediction = model.predict(new_transaction)

if prediction[0] == 1:
    print("Fraud Transaction")
else:
    print("Genuine Transaction")

# %%
probability = model.predict_proba(new_transaction)

print(probability)

# %%
test_df = pd.read_csv(r"C:\Users\anil kumar singh\OneDrive\Desktop\Vendor Performance\test_transactions_1000.csv")
prediction = model.predict(test_df)
test_df["Prediction"]= prediction
test_df["Result"] = test_df["Prediction"].map({0: "Genuine", 1: "Fraud"})
print(test_df[["Amount", "Result"]])

# %%
sample = df.sample(10, random_state=42)

actual = sample["Class"]

features = sample.drop("Class", axis=1)

predicted = model.predict(features)

result = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted
})

print(result)

# %%
import joblib

joblib.dump(model, "fraud_detection_model.pkl")

print("Model saved successfully!")

# %%
model = joblib.load("fraud_detection_model.pkl")

# %%
test_df["Prediction"] = prediction

test_df["Result"] = test_df["Prediction"].map({
    0: "Genuine",
    1: "Fraud"
})

test_df.to_csv("predicted_transactions.csv", index=False)

# %%



