# %%
import pandas as pd
import numpy as np
# import os
# import sys
# import argparse
df = pd.read_csv(r"C:\Users\anil kumar singh\OneDrive\Desktop\Vendor Performance\credit_card.csv")
print(df.head())
fraud_df = df[df["Class"] == 1]
print(fraud_df)

# %%
print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# %%
# Count duplicate rows
duplicates = df.duplicated().sum()

print("\nDuplicate Rows:")
print(duplicates)


# %%
# Show duplicate rows
print("\nDuplicate Data:")
print(df[df.duplicated()])

# %%
# Remove duplicates (optional)
df = df.drop_duplicates()

# %%
#  INCORRECT DATA TYPES
print("\nData Types:")
print(df.dtypes)

# %%
# Select numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\nColumn: {col}")
    print(f"Number of Outliers: {len(outliers)}")

# %%
#  REMOVE OUTLIERS
for col in numerical_cols:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\nDataset Shape After Removing Outliers:")
print(df.shape)

# %%
print("\nCleaned Dataset Preview:")
print(df.head())

# %%
fraud_df = df[df["Class"] == 1]
print(fraud_df)

# %%
df = pd.read_csv(r"C:\Users\anil kumar singh\OneDrive\Desktop\Vendor Performance\credit_card.csv")
fraud_df = df[df["Class"] == 1]
print(fraud_df);

# %%
df.to_csv("creditcard_cleaned.csv", index=False)

print("Cleaned data saved as creditcard_cleaned.csv")

# %%



