import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("credit_scoring_pre.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

df = df.drop_duplicates()
df = df.dropna()
categorical_columns = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop("total_income", axis=1)
y = df["total_income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

model_tree = DecisionTreeRegressor(random_state=42)
model_tree.fit(X_train_processed, y_train)
y_pred_tree = model_tree.predict(X_test_processed)
r2_tree = r2_score(y_test, y_pred_tree)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_tree)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Decision Tree: Predicted vs Actual")
plt.grid(True)
plt.show()

model_nn = Sequential()
model_nn.add(Dense(64, activation='relu', input_dim=X_train_processed.shape[1]))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(1))
model_nn.compile(optimizer='adam', loss='mse')

model_nn.fit(X_train_processed, y_train, epochs=50, batch_size=32, verbose=0)
y_pred_nn = model_nn.predict(X_test_processed)
r2_nn = r2_score(y_test, y_pred_nn)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_nn)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Neural Network: Actual vs Predicted")
plt.grid(True)
plt.show()

print(r2_tree)
print(r2_nn)
