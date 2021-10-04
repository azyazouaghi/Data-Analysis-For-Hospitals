import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 15)

general = pd.read_csv("test/general.csv")
prenatal = pd.read_csv("test/prenatal.csv")
sports = pd.read_csv("test/sports.csv")

# Changing columns names
prenatal.columns = list(general.columns)
sports.columns = list(general.columns)

# Merging datasets into one

dataset = pd.concat([general, prenatal, sports], axis=0, ignore_index=True)

# Deleting Unnamed:0 column
dataset.drop(columns=['Unnamed: 0'], inplace=True)

# Deleting empty rows
dataset.dropna(axis=0, how="all", inplace=True)

# Setting gender values to f and m
dataset['gender'].replace({'female': 'f', 'woman': 'f', 'male': 'm', 'man': 'm', np.nan: 'f'}, None, inplace=True)

# Filling NaN values with 0's
dataset.fillna(0, inplace=True)

"""
-- 1st question: hospital with the highest number of patients
"""
# Grouping rows by hospital
groups = dataset.groupby('hospital')
# Selecting the index of the max value (name of the hospital)
answer = groups['hospital'].count().idxmax()
print(f'The answer to the 1st question is {answer}')

"""
-- 2nd question:  share of the patients in the general hospital suffering from stomach-related issues
"""
# Creating a pivot table
pivot = dataset.pivot_table(index='hospital', columns='diagnosis', aggfunc='count')
answer = (pivot.loc['general', 'xray']['stomach'] / pivot.loc['general', 'xray'].sum()).round(3)
print(f'The answer to the 2nd question is {answer}')

"""
-- 3rd question:  share of the patients in the sports hospital suffering from dislocation-related issues
"""
# Creating a pivot table
pivot = dataset.pivot_table(index='hospital', columns='diagnosis', aggfunc='count')
answer = (pivot.loc['sports', 'xray']['dislocation'] / pivot.loc['sports', 'xray'].sum()).round(3)
print(f'The answer to the 3rd question is {answer}')

"""
-- 4th question:  difference in the median ages of the patients in the general and sports hospitals
"""
# Creating a pivot table
pivot = dataset.pivot_table(index='hospital', values='age', aggfunc='median')
answer = pivot.loc['general']['age'] - pivot.loc['sports']['age']
print(f'The answer to the 4th question is {int(answer)}')

"""
-- 5th question:  hospital in which the blood test was taken the most often
"""
# Creating a pivot table
pivot = dataset.pivot_table(index='hospital', columns='blood_test', aggfunc='count')
answer_1 = pivot['age']['t'].max()
answer_2 = pivot['age']['t'].idxmax()
print(f'The answer to the 5th question is {answer_2}, {int(answer_1)} blood tests')

"""
-- 1st question: The most common age of a patient among all hospitals
"""
dataset['age'].value_counts().plot(kind='hist', bins=5)
plt.show()
print("The answer to the 1st question: 15 - 35")

"""
-- 2nd question: The most common diagnosis among patients in all hospitals
"""
dataset['diagnosis'].value_counts().plot(kind='pie')
plt.show()
print("The answer to the 2nd question: pregnancy")

"""
-- 3rd question: The main reason for the gap in values
"""
df = dataset[['hospital', 'height']]
ax = sns.violinplot(x='hospital', y='height', data=df)
plt.show()
print(
    "The answer to the 3rd question: It's because both general and prenatal hospitals are using meters as measurement unit. "
    "However, sports hospital is using centimeters. "
    "The two peaks are the first quartile — 1.5 IQR and the third quartile + 1.5 IQR. They are used to detect outliers.")
