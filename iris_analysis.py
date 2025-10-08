# Data Analysis and Visualization on Iris Dataset
# Author: [Your Name]
# Date: [Insert Date]

# ============================
# Task 1: Load and Explore the Dataset
# ============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Handle errors during file loading
try:
    # Load Iris dataset directly from seaborn
    iris = sns.load_dataset('iris')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: File not found. Please check your dataset path.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Display first few rows
print("\n🔹 First 5 rows of the dataset:")
print(iris.head())

# Check data structure
print("\n🔹 Dataset Info:")
print(iris.info())

# Check for missing values
print("\n🔹 Missing Values:")
print(iris.isnull().sum())

# Clean data (drop missing values if any)
iris = iris.dropna()

# ============================
# Task 2: Basic Data Analysis
# ============================

# Basic statistics
print("\n🔹 Basic Statistics:")
print(iris.describe())

# Grouping: Compute mean of numerical columns for each species
grouped_means = iris.groupby('species').mean(numeric_only=True)
print("\n🔹 Mean values by Species:")
print(grouped_means)

# Identify patterns or findings
print("\n🔹 Insights:")
print("- Iris setosa tends to have smaller petal and sepal sizes.")
print("- Iris virginica generally has the largest petal and sepal sizes.")
print("- Iris versicolor lies in between.")

# ============================
# Task 3: Data Visualization
# ============================

# 1️⃣ Line Chart (Trend Over Index)
plt.figure(figsize=(8,5))
plt.plot(iris.index, iris['sepal_length'], label='Sepal Length', color='blue')
plt.title('Trend of Sepal Length Over Observations')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2️⃣ Bar Chart — Average Petal Length per Species
plt.figure(figsize=(7,5))
sns.barplot(x='species', y='petal_length', data=iris, palette='pastel', estimator='mean')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3️⃣ Histogram — Distribution of Sepal Width
plt.figure(figsize=(7,5))
plt.hist(iris['sepal_width'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4️⃣ Scatter Plot — Sepal Length vs Petal Length
plt.figure(figsize=(7,5))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris, palette='cool')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# ============================
# Summary of Findings
# ============================
print("\n🧩 Summary of Findings:")
print("- The Iris dataset contains no missing values and 3 unique species.")
print("- Iris setosa flowers are generally smaller compared to others.")
print("- Petal length and sepal length show a strong positive correlation.")
print("- Data distributions are mostly normal, with slight variations between species.")
