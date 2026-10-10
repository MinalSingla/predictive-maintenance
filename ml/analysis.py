import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/ai4i2020.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nMachine failure counts:")
print(df["Machine failure"].value_counts())

print("\nMachine failure percentages:")
print(
    df["Machine failure"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

df["Machine failure"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

df["Rotational speed [rpm]"].hist(bins=30)

plt.title("Distribution of Rotational Speed")
plt.xlabel("Rotational Speed (rpm)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

drop_columns = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

prepared_df = df.drop(columns=drop_columns)

prepared_df = prepared_df.rename(columns={
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure"
})

print("\nPrepared dataset:")
print(prepared_df.head())

print("\nPrepared dataset shape:")
print(prepared_df.shape)

print("\nPrepared dataset columns:")
print(prepared_df.columns.tolist())

prepared_df.to_csv("data/prepared_data.csv", index=False)

print("\nPrepared dataset saved successfully.")