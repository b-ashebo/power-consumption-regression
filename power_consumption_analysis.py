"""
Power Consumption Analysis & Regression Modeling

Explores a weather/environmental dataset and predicts the average power
consumption across three city zones using Linear, Ridge, Lasso, and
Elastic Net regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load data
df = pd.read_csv("powerconsumption.csv")
print(df.head())
print("Shape:", df.shape)

zone_cols = [
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3",
]

# Target: average consumption across the three zones
df["PowerConsumption_Zone"] = df[zone_cols].mean(axis=1)


# Inspect & clean
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe())

df["Datetime"] = pd.to_datetime(df["Datetime"], format="%m/%d/%Y %H:%M")


# Exploratory plots
# Boxplot of the three zones
df[zone_cols].plot(kind="box", figsize=(8, 5), title="Boxplot of Selected Features")
plt.ylabel("Values")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Temperature over time
plt.figure(figsize=(10, 4))
plt.plot(df["Datetime"], df["Temperature"], color="blue")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Wind speed over time
plt.figure(figsize=(10, 4))
plt.plot(df["Datetime"], df["WindSpeed"], color="blue")
plt.xlabel("Date Time")
plt.ylabel("WindSpeed")
plt.tight_layout()
plt.show()

# Diffuse flows over time
plt.figure(figsize=(10, 4))
plt.plot(df["Datetime"], df["GeneralDiffuseFlows"], color="blue", label="GeneralDiffuseFlows")
plt.plot(df["Datetime"], df["DiffuseFlows"], color="red", label="DiffuseFlows")
plt.xlabel("Date Time")
plt.ylabel("Flows")
plt.legend()
plt.tight_layout()
plt.show()

# Humidity over time
plt.figure(figsize=(10, 4))
plt.plot(df["Datetime"], df["Humidity"], color="blue")
plt.xlabel("Date Time")
plt.ylabel("Humidity")
plt.tight_layout()
plt.show()

# Power consumption per zone over time
plt.figure(figsize=(10, 4))
for col, color in zip(zone_cols, ["blue", "red", "green"]):
    plt.plot(df["Datetime"], df[col], color=color, label=col.replace("PowerConsumption_", ""))
plt.xlabel("Date Time")
plt.ylabel("Power Consumption")
plt.legend()
plt.tight_layout()
plt.show()

# Cumulative power consumption over temperature
area = df.groupby("Temperature")[zone_cols].mean().sort_index()
plt.figure(figsize=(8, 5))
plt.stackplot(
    area.index,
    area.T,
    labels=[c.replace("PowerConsumption_", "") for c in zone_cols],
    alpha=0.6,
)
plt.legend()
plt.title("Cumulative Power Consumption Over Temperature")
plt.xlabel("Temperature")
plt.ylabel("Power Consumption")
plt.tight_layout()
plt.show()

# Distribution of all numeric features
dist_cols = [
    "Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows",
    "PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3",
    "PowerConsumption_Zone",
]
df[dist_cols].hist(bins=30, figsize=(12, 10))
plt.suptitle("Data Distribution")
plt.tight_layout()
plt.show()


# Correlation analysis
cor_matrix = df[dist_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

cor_pairs = cor_matrix.stack().reset_index()
cor_pairs.columns = ["Var1", "Var2", "Correlation"]
cor_pairs = cor_pairs[cor_pairs["Var1"] != cor_pairs["Var2"]]
cor_pairs["abs_corr"] = cor_pairs["Correlation"].abs()

high_corr = cor_pairs[(cor_pairs["abs_corr"] > 0.6) & (cor_pairs["abs_corr"] < 1)] \
    .sort_values("abs_corr", ascending=False)
print("\nHigh Correlation Pairs:\n", high_corr)

low_corr = cor_pairs[cor_pairs["abs_corr"] < 0.11].sort_values("abs_corr")
print("\nLow Correlation Pairs:\n", low_corr)


# Standardization & normalization
numerical_columns = dist_cols

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)
print("\nStandardized data:\n", df_scaled.head())

minmax = MinMaxScaler()
df_normalized = pd.DataFrame(minmax.fit_transform(df[numerical_columns]), columns=numerical_columns)
print("\nNormalized data:\n", df_normalized.head())

df_scaled.hist(bins=30, figsize=(12, 10), color="blue")
plt.suptitle("Standardized Distributions")
plt.tight_layout()
plt.show()

df_normalized.hist(bins=30, figsize=(12, 10), color="blue")
plt.suptitle("Normalized Distributions")
plt.tight_layout()
plt.show()


# Train/test split
X = df_scaled
y = df["PowerConsumption_Zone"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear regression summary 
ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print("\n", ols.summary())



# Fit & evaluate all models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": RidgeCV(alphas=np.logspace(-3, 3, 100)),
    "Lasso Regression": LassoCV(cv=10, random_state=42),
    "Elastic Net": ElasticNetCV(l1_ratio=0.5, cv=10, random_state=42),
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"{name:<20} MSE: {mse:.6f}  MAE: {mae:.6f}")

residuals = {name: y_test - pred for name, pred in predictions.items()}


# Residual diagnostics
# QQ plots
for name, res in residuals.items():
    plt.figure()
    stats.probplot(res, dist="norm", plot=plt)
    plt.title(f"QQ Plot - {name}")
    plt.tight_layout()
    plt.show()

# Residuals vs fitted (predicted) values
for name, pred in predictions.items():
    plt.figure()
    plt.scatter(pred, residuals[name], color="blue", s=10)
    plt.axhline(0, color="red")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Fitted - {name}")
    plt.tight_layout()
    plt.show()

# Histograms of residuals
for name, res in residuals.items():
    plt.figure()
    plt.hist(res, color="blue", edgecolor="darkblue")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Residuals - {name}")
    plt.tight_layout()
    plt.show()
