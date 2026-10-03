import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

df["Region"] = df["Region"].fillna("Unknown")

print("\nDescriptive statistics:\n", df[["Units_Sold","Unit_Price","Revenue"]].describe())
summary = df.groupby("Product").agg(
    Total_Units=("Units_Sold","sum"),
    Total_Revenue=("Revenue","sum"),
    Average_Revenue=("Revenue","mean")
).sort_values("Total_Revenue", ascending=False)
print("\nProduct summary:\n", summary.round(2))

print("\nRevenue by region:\n", df.groupby("Region")["Revenue"].sum().sort_values(ascending=False).round(2))
print("\nCorrelation:\n", df[["Units_Sold","Unit_Price","Revenue"]].corr().round(2))

summary["Total_Revenue"].plot(kind="bar", figsize=(8,5))
plt.title("Total Revenue by Product")
plt.xlabel("Product"); plt.ylabel("Revenue"); plt.tight_layout()
plt.savefig("revenue_by_product.png", dpi=150); plt.show()

df.groupby("Date")["Revenue"].sum().plot(figsize=(8,5))
plt.title("Daily Revenue Trend")
plt.xlabel("Date"); plt.ylabel("Revenue"); plt.tight_layout()
plt.savefig("daily_revenue_trend.png", dpi=150); plt.show()
