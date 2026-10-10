import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

pumpkins = pd.read_csv('pumpkins.csv')

print(pumpkins.head())
print('df iniziale')
print(' ')


pumpkins_due = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]

print(pumpkins_due.head())
print('df solo con bushel')
print(' ')

columns_to_select = ['Package', 'Variety', 'City Name', 'Low Price', 'High Price', 'Date']
pumpkins_tre = pumpkins.loc[:, columns_to_select]

print(pumpkins_tre.head())
print('df con colonne selezionate')
print(' ')

price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

month = pd.DatetimeIndex(pumpkins['Date']).month
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)

new_pumpkins = pd.DataFrame(
    {'Month': month, 
     'DayOfYear' : day_of_year, 
     'Variety': pumpkins['Variety'], 
     'City': pumpkins['City Name'], 
     'Package': pumpkins['Package'], 
     'Low Price': pumpkins['Low Price'],
     'High Price': pumpkins['High Price'], 
     'Price': price})

new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/1.1
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price*2
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE'].copy()

print(pie_pumpkins.head())
print('df con nuove colonne e prezzo medio')
print(' ')

