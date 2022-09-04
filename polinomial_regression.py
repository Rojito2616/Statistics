# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:48:44 2022

@author: Julio Rojas
"""

# Regresión polinómica 

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Book1.csv')
dataset.drop(['Hemoglobina Glucosilada', 'T2D= diabetes tipo 2. \n'], axis=1, inplace=True)
dataset.dropna(inplace = True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la regresión polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualización de los resultados del Modelo Lineal
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Glucosa en ayunas (mg/dl)")
plt.ylabel("Glucosa pos carga (mg/dl) 2h > 75 g")
plt.show()

# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Glucosa en ayunas (mg/dl)")
plt.ylabel("Glucosa pos carga (mg/dl) 2h > 75 g")
plt.show()
print("y = ", lin_reg.coef_, "X", "+", lin_reg.intercept_)

#%%
dataset_2 = pd.read_csv('Book1.csv')
dataset_2.drop(['Hemoglobina Glucosilada', 'T2D= diabetes tipo 2. \n'], axis=1, inplace=True)
NaN = dataset_2[pd.isnull(dataset_2).any(axis=1)]
remove_34=NaN.drop(NaN.index[[33]])
X_test = remove_34.iloc[:, :-1].values
# Predecir el conjunto de test
y_pred = lin_reg.predict(X_test)

#%%

"""
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))
print('w = ' + str(lin_reg_2.coef_) + ', b = ' + str(lin_reg_2.intercept_))
"""
#%%
#Sustituir NaN's
df = pd.read_csv('Book1.csv')
df.drop(df.index[[50]], inplace=True)
X_test_df = pd.DataFrame(X_test)
X_test_df.rename(columns = {0 : 'Glucosa en ayunas (mg/dl)'}, inplace = True)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.rename(columns = {0 : 'glucosa pos carga (mg/dl) 2h > 75 g '}, inplace = True)
y_pred_df_int=y_pred_df.astype(int)
Results = pd.concat([X_test_df,y_pred_df_int], axis=1)


#%%
nuevo_dict = dict(zip(Results["Glucosa en ayunas (mg/dl)"], Results["glucosa pos carga (mg/dl) 2h > 75 g "]))
#%%
df['glucosa pos carga (mg/dl) 2h > 75 g ']= df['glucosa pos carga (mg/dl) 2h > 75 g '].fillna(df['Glucosa en ayunas (mg/dl)'].map(nuevo_dict))
#%%
df['Hemoglobina Glucosilada'] = df['Hemoglobina Glucosilada'].fillna(df['Hemoglobina Glucosilada'].mean())
df['Hemoglobina Glucosilada'] = df['Hemoglobina Glucosilada'].round(decimals=1)
#%%
"""
Gráficas
"""

df.plot.scatter(y="glucosa pos carga (mg/dl) 2h > 75 g ",x="Glucosa en ayunas (mg/dl)")
pd.plotting.scatter_matrix(df) 
#%%
corre = pd.DataFrame(df.corr())

serie = df["T2D= diabetes tipo 2. \n"].value_counts()
serie.plot.pie(autopct='%1.1f%%')
serie.plot.bar()

#%%
# Gráficas de Glucosa en ayunas (mg/dl)
df["Glucosa en ayunas (mg/dl)"].plot.hist()
df["Glucosa en ayunas (mg/dl)"].plot.density()
df["Glucosa en ayunas (mg/dl)"].plot.box()
#%%
# glucosa pos carga (mg/dl) 2h > 75 g           
df["glucosa pos carga (mg/dl) 2h > 75 g "].plot.hist()
df["glucosa pos carga (mg/dl) 2h > 75 g "].plot.density()
df["glucosa pos carga (mg/dl) 2h > 75 g "].plot.box()

#%%
# Hemoglobina Glucosilada           
df["Hemoglobina Glucosilada"].plot.hist()
df["Hemoglobina Glucosilada"].plot.density()
df["Hemoglobina Glucosilada"].plot.box()

                                

                                

                                

                                

                                

                                

                                

                                

                        
