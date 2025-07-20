# preprocessing_rf.py

# Heading: Import libraries
import pandas as pd

# Heading: Load dataset
df = pd.read_csv("diabetes.csv")

# Heading: Replace 0s in medically invalid columns with NaN
invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[invalid_cols] = df[invalid_cols].replace(0, pd.NA)

# Heading: Fill missing values with median of each column
for col in invalid_cols:
    df[col] = df[col].fillna(df[col].median())

# Heading: Save the cleaned dataset
df.to_csv("diabetes_clean_rf.csv", index=False)
print("Cleaned data saved as 'diabetes_clean_rf.csv'")
