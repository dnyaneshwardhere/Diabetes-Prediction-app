import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# Load Dataset
df = pd.read_csv("diabetes.csv")

# Convert All Columns to Numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic Information
print("Shape:", df.shape)
print("Data types:\n", df.dtypes)
print("\nClass Distribution:\n", df['Outcome'].value_counts(normalize=True))

# Replace Invalid Zeros with NaN
cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

# Fill Missing Values with Median
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].fillna(df[cols_with_invalid_zeros].median())

# Boxplots for Outlier Detection
plt.figure(figsize=(15, 8))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(y=col, data=df)
plt.tight_layout()
plt.suptitle("Boxplots to Detect Outliers", y=1.02)
plt.show()

# Class Distribution Plot
sns.countplot(x='Outcome', data=df)
plt.title("Class Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Violin Plots for Feature Distributions
df_melted = df.drop('Outcome', axis=1).melt()
plt.figure(figsize=(12, 8))
sns.violinplot(x='variable', y='value', data=df_melted, inner='quartile')
plt.xticks(rotation=45)
plt.title("Feature Distributions (Violin Plots)")
plt.tight_layout()
plt.show()

# Feature Importance using Mutual Information
X = df.drop('Outcome', axis=1)
y = df['Outcome']
mi_scores = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores}).sort_values(by='MI Score', ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=mi_df, x='MI Score', y='Feature', palette='viridis')
plt.title("Feature Importance (Mutual Information)")
plt.show()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Save Processed Dataset
processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
processed_df.to_csv("processed_diabetes_data.csv", index=False)
print("\nProcessed data saved as 'processed_diabetes_data.csv'")
