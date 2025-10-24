# 观察数据集的统计特征
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('bike_sharing_hour.csv')
print("--- Data Head ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())
print("\n--- Descriptive Statistics ---")
print(df.describe())
print("\n--- Null Values Count ---")
print(df.isnull().sum())

# --- 相关性分析 ---
# 仅选择数值类型的列来计算相关性
numeric_df = df.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()
print("\n--- Correlation Matrix ---")
print(corr_matrix)

# 使用热力图可视化相关性矩阵
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features', fontsize=16)
plt.show()


# --- 数据分布观察 ---

# 1. 观察目标变量 'cnt' 的分布
plt.figure(figsize=(12, 6))
sns.histplot(df['cnt'], bins=50, kde=True)
plt.title('Distribution of Total Bike Rentals (cnt)', fontsize=16)
plt.xlabel('Total Rentals (cnt)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# 2. 观察关键数值特征的分布
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
df[numeric_features].hist(bins=30, figsize=(15, 10), layout=(2, 2))
plt.suptitle('Distribution of Key Numeric Features', fontsize=16)
plt.show()

# 3. 观察季节(season)这个类别特征的分布
plt.figure(figsize=(10, 6))
sns.countplot(x='season', data=df)
plt.title('Bike Rentals Count by Season', fontsize=16)
# 根据数据集说明，1:春, 2:夏, 3:秋, 4:冬
plt.xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'])
plt.xlabel('Season', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# 4. 观察不同小时(hr)和工作日(workingday)下的租车数量分布
plt.figure(figsize=(14, 7))
sns.pointplot(data=df, x='hr', y='cnt', hue='workingday')
plt.title('Average Bike Rentals by Hour and Working Day', fontsize=16)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Average Total Rentals', fontsize=12)
plt.legend(title='Working Day', labels=['Holiday/Weekend', 'Working Day'])
plt.grid(True)
plt.show()