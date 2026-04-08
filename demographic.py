#Tools
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Loads data
df = pd.read_csv("demographic.txt")

#Convert education to numeric
df["education"] = df["education"].map({
    "HighSchool": 0,
    "College": 1,
    "Graduate": 2
})

#Correlation analysis 
corr_matrix = df.corr()
print(corr_matrix.round(2))

#Define predictors and outcomes
X = df[["pantry_use", "age", "education","income", "BMI", "distance_to_pantry"]]
Y = df["BP"]

#Add constant (intercept)
X = sm.add_constant(X)

#Fit model
model = sm.OLS(Y,X).fit()

#Show Results multiple linear regression
print(model.summary())

# Accuracy
print("Accuracy:", accuracy_score(Y, df["BP"]))

# Confusion matrix
print(confusion_matrix(Y, df["BP"]))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

