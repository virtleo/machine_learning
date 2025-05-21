import numpy as np
import pandas as pd

df = pd.read_csv('./week1/sales.csv')
#print(df.head())
df['Date'] = pd.to_datetime(df['Date'])

# 填充 Quantity 列缺失值：用相同产品的历史销量中位数替换
df['Quantity'] = df.groupby('Product')['Quantity'].transform(lambda x: x.fillna(x.median()))

# (2) 计算每行的销售额（Price * Quantity）
df['TotalSales'] = df['Price'] * df['Quantity']

# 按产品计算总销售额和平均单价
product_summary = df.groupby('Product').agg(Total_Sales=('TotalSales', 'sum'),Average_Price=('Price', 'mean')).reset_index()

# (3) 按月份统计所有产品的总销售额
# 提取月份信息（使用 Period 格式，例如 "2023-01"）
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['TotalSales'].sum().reset_index()

# 输出结果
print("每个产品的总销售额和平均单价：")
print(product_summary)
print("\n各月份总销售额：")
print(monthly_sales)
print("\n修改后的sales.csv数据：")
print(df.head())