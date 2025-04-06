import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold

# -----------------------------------
# Read data from file
# -----------------------------------
print('Loading file ...')
column_names = ['Obser', 'Y1', 'Y2', 'X1', 'X2', 'X3', 'X4', 'X5']
df_research = pd.read_csv("../data/ANN_Data_2Y_5X_4760_Observationtxt.txt", sep='\t',header=0, names=column_names, index_col=False)
# print(df_research)

# -----------------------------------
# Delete some unnecessary columns
# -----------------------------------
df_research = df_research.drop(columns=['Obser'])
# print(df_research.columns)
df_research['X5'] = df_research['X5'].astype(str).str.extract(r'^(\d+\.?\d*)').astype(float)
# print(df_research.iloc[2,:])
# print(df_research.head())

# Create X and Y
X = df_research[['X1', 'X2', 'X3', 'X4', 'X5']]
y = df_research[['Y1', 'Y2']]

X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Write to csv file
print('Convert dataset to .csv file of input X and output y')
X.to_csv('../data/X.csv', index = False)
y.to_csv('../data/y.csv', index = False)

X_test.to_csv('../data/X_test.csv', index = False)
y_test.to_csv('../data/y_test.csv', index = False)