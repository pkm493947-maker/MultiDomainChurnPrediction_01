import pandas as pd

print("📌 Checking dataset columns...\n")

telecom = pd.read_csv("../datasets/telecom.csv")
banking = pd.read_csv("../datasets/banking.csv")
ecommerce = pd.read_csv("../datasets/ecommerce.csv")

print("Telecom Columns:\n", telecom.columns.tolist(), "\n")
print("Banking Columns:\n", banking.columns.tolist(), "\n")
print("Ecommerce Columns:\n", ecommerce.columns.tolist(), "\n")