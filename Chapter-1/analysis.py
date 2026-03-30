import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the survey data
print("Loading survey data...")
df = pd.read_excel('SurveyData.xlsx')

# Display basic information about the dataset
print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "="*80)
print("FIRST FEW ROWS")
print("="*80)
print(df.head(10))

print("\n" + "="*80)
print("DATA TYPES")
print("="*80)
print(df.dtypes)

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(df.isnull().sum())

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)
print(df.describe())

print("\n" + "="*80)
print("UNIQUE VALUES PER COLUMN")
print("="*80)
for col in df.columns:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique} unique values")
    if n_unique <= 10:
        print(f"  Values: {sorted(df[col].dropna().unique())}")
