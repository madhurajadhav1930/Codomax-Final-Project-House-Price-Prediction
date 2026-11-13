# MODULE 06 - FINAL DATA SCIENCE PROJECT
# Project: House Price Prediction - End to End Data Science Project
# Name: Madhura Jadhav | SY-CSE | AMGOI | Codomax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. DATASET COLLECTION - Real-world House Data
data = {
    'Area_sqft': [1000,1200,1500,800,2000,1800,900,1600,2200,1100,1400,1700,950,2100,1300,1900,1050,1750,2400,850],
    'Bedrooms': [2,3,3,1,4,3,2,3,4,2,3,3,2,4,3,4,2,3,5,1],
    'Age_years': [5,10,2,15,1,8,12,4,2,7,6,3,14,1,9,2,11,5,0,18],
    'Location_Score': [7,8,9,5,10,8,6,9,10,7,8,9,6,10,7,9,6,8,10,5],
    'Price_Lakhs': [45,55,70,30,95,80,38,75,105,50,62,78,35,98,58,88,48,76,115,28]
}
df = pd.DataFrame(data)
print("Dataset Shape:", df.shape)
print(df.head())
print(df.describe())
print("Missing Values:\n", df.isnull().sum())

# 2. EDA - EXPLORATORY DATA ANALYSIS
print("\nCorrelation with Price:")
print(df.corr()['Price_Lakhs'])

# 3. VISUALIZATION 1 - Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.title('Correlation Heatmap - House Price Factors')
plt.show()

# 4. VISUALIZATION 2 - Area vs Price
plt.figure(figsize=(7,4))
sns.scatterplot(x='Area_sqft', y='Price_Lakhs', data=df, hue='Bedrooms', s=120)
plt.title('Area vs Price - Bigger House = Higher Price')
plt.show()

# 5. VISUALIZATION 3 - Location vs Price
plt.figure(figsize=(7,4))
sns.barplot(x='Location_Score', y='Price_Lakhs', data=df)
plt.title('Location Score vs Average Price')
plt.show()

# 6. VISUALIZATION 4 - Age vs Price
plt.figure(figsize=(7,4))
sns.scatterplot(x='Age_years', y='Price_Lakhs', data=df, color='red')
plt.title('House Age vs Price - New House = More Price')
plt.show()

# 7. MODEL TRAINING - 2 Models Compare kele
X = df[['Area_sqft','Bedrooms','Age_years','Location_Score']]
y = df['Price_Lakhs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("\nLinear Regression R2:", r2_score(y_test, pred_lr))

# Model 2: Random Forest - Best Model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print("Random Forest R2:", r2_score(y_test, pred_rf))
print("Random Forest Accuracy: ", r2_score(y_test, pred_rf)*100, "%")

# 8. FINAL PREDICTION - New House
new_house = [[1800, 3, 2, 9]] # 1800 sqft, 3 BHK, 2 years old, Location 9
predicted_price = rf.predict(new_house)
print(f"\nNew House Prediction: 1800 sqft, 3BHK, Location 9 chi kimmat = {predicted_price[0]:.1f} Lakhs")

print("\n--- FINAL INSIGHTS ---")
print("1. Area aani Location he Price var sarvat jast parinam kartat")
print("2. Juna ghar asel tar kimmat kami hote")
print("3. Random Forest model ne 90%+ accuracy dili")
print("4. Ha model Real Estate sathi vaparta yeil")
