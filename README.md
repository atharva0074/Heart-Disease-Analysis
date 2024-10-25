# heart_disease_analysis.py

# 1. Import Packages and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/content/heart.csv')

# 2. Display Top 5 Rows of the dataset
print("Top 5 rows of the dataset:")
print(data.head())

# 3. Check the last 5 Rows of the dataset
print("Last 5 rows of the dataset:")
print(data.tail())

# 4. Find shape of Our Dataset
print("Number of Rows:", data.shape[0])
print("Number of Columns:", data.shape[1])

# 5. Get information about our dataset
data.info()

# 6. Check Null values in the dataset
print("Null values in the dataset:")
print(data.isnull().sum())

# 7. Check for Duplicate data and drop them
data_dup = data.duplicated().any()
print("Are there any duplicates?", data_dup)

data = data.drop_duplicates()
print("Shape after dropping duplicates:", data.shape)

# 8. Get overall statistics about the dataset
print("Overall statistics:")
print(data.describe())

# 9. Draw Correlation Matrix
plt.figure(figsize=(15, 6))
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# 10. Count of people with and without heart disease
print("Count of heart disease cases:")
print(data['target'].value_counts())
sns.countplot(x='target', data=data)
plt.title("Heart Disease Count")
plt.show()

# 11. Count of male & female
print("Count of males and females:")
print(data['sex'].value_counts())
sns.countplot(x='sex', data=data)
plt.title("Gender Distribution")
plt.show()

# 12. Gender distribution according to the target variable
sns.countplot(x='sex', hue='target', data=data)
plt.xticks([0, 1], ['Female', 'Male'])
plt.legend(labels=['No-Disease', 'Disease'])
plt.title("Gender Distribution by Heart Disease")
plt.show()

# 13. Check age distribution in the dataset
sns.histplot(data['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# 14. Check chest pain type
sns.countplot(x='cp', data=data)
plt.xticks([0, 1, 2, 3], ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
plt.title("Chest Pain Types")
plt.show()

# 15. Chest pain distribution as per target variable
sns.countplot(x='cp', hue='target', data=data)
plt.xticks([0, 1, 2, 3], ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
plt.title("Chest Pain Distribution by Heart Disease")
plt.show()

# 16. Fasting blood sugar distribution according to the target variable
sns.countplot(x='fbs', hue='target', data=data)
plt.title("Fasting Blood Sugar Distribution by Heart Disease")
plt.show()

# 17. Resting blood pressure distribution
data['trestbps'].plot(kind='hist')
plt.title("Resting Blood Pressure Distribution")
plt.show()

# 18. Compare Resting blood pressure as per sex column
g = sns.FacetGrid(data, hue='sex', aspect=4)
g.map(sns.kdeplot, 'trestbps', shade=True)
plt.legend(labels=['Male', 'Female'])
plt.title("Resting Blood Pressure by Gender")
plt.show()

# 19. Show distribution of serum cholesterol
data['chol'].plot(kind='hist')
plt.title("Serum Cholesterol Distribution")
plt.show()

# 20. Plot continuous variables
data.hist(figsize=(15, 10))
plt.suptitle("Distribution of Continuous Variables")
plt.show()
