import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Read file
print('Loading file ...')
column_names = ['Obser', 'Y1', 'Y2', 'X1', 'X2', 'X3', 'X4', 'X5']
df_research = pd.read_csv("ANN_Data_2Y_5X_4760_Observationtxt.txt", sep='\t',header=0, names=column_names, index_col=False)
# print(df_research)

# Delete 'Obser' because it is a row number
df_research = df_research.drop(columns=['Obser'])
# print(df_research.columns)
df_research['X5'] = df_research['X5'].astype(str).str.extract(r'^(\d+\.?\d*)').astype(float)
# print(df_research.iloc[2,:])

# Check the first few rows
# print(df_research.head())

# create X and Y
X = df_research[['X1', 'X2', 'X3', 'X4', 'X5']]
y = df_research[['Y1', 'Y2']]

# write to csv file
print('Convert dataset to .csv file of input X and output y')
X.to_csv('X.csv')
y.to_csv('y.csv')