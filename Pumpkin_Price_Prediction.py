import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

pumpkins = pd.read_csv('pumpkins.csv')


pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]

columns_to_select = ['Package', 'Variety', 'City Name', 'Low Price', 'High Price', 'Date']
pumpkins = pumpkins.loc[:, columns_to_select]

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

new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/1.1 # aggiungi ad obsidian
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price*2
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE'].copy()


pie_pumpkins.dropna(inplace=True)

X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']

encoded_city = pd.get_dummies(pie_pumpkins['City'], drop_first=True, dtype=int) 
X_mod = pd.concat([pie_pumpkins['DayOfYear'], encoded_city], axis=1) #axis=1 affianca per colonne anziche impilare per righe come axis=0 (default)
y_mod = pie_pumpkins['Price']

scores = {'linear_regression_standard':[], 'polynomial_regression_standard':[], 'linear_regression_mod':[], 'polynomial_regression_mod':[]
}

for random_state in range(40):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    scores['linear_regression_standard'].append(model.score(X_test, y_test))
    
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    scores['polynomial_regression_standard'].append(model.score(X_test_poly, y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X_mod, y_mod, test_size=0.33, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    scores['linear_regression_mod'].append(model.score(X_test, y_test))
    
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    scores['polynomial_regression_mod'].append(model.score(X_test_poly, y_test))

for name, values in scores.items():
    std = np.std(values)
    mean = np.mean(values)
    print(f'Model: {name} | Mean: {mean:.3f} | Std: {std:.3f}')
print(scores['linear_regression_standard'][0])
print(scores['polynomial_regression_standard'][0])
print(scores['linear_regression_mod'][0])
print(scores['polynomial_regression_mod'][0])