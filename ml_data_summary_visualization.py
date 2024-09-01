# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset from scikit-learn
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target

# Encode target labels to make them more readable
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# Mapping target names for better understanding
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Summary Statistics
print("Summary Statistics of the Dataset:")
print(df.describe())

# Checking for missing values
print("\nMissing Values in the Dataset:")
print(df.isnull().sum())

# Data Visualization
# Pairplot of all features
sns.pairplot(df, hue='species')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Boxplot to show the distribution of features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('Boxplot of Features')
plt.xticks(rotation=45)
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Iris Dataset')
plt.show()

# Histogram of each feature
df.hist(figsize=(12, 8), bins=15, color='skyblue', edgecolor='black')
plt.suptitle('Histogram of Each Feature', y=1.02)
plt.show()
