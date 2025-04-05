import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'C:\Users\Gopi Reddy\NIT7PM\Apr\emp_sal.csv')

# linear model  -- linear algor ( degree - 1)
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# linear regression visualizaton 
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x),color = 'blue')
plt.title('Linear regression model(Linear Regression)')
plt.xlabel('Positiion Level')
plt.ylabel('Salary')
plt.show()

m = lin_reg.coef_
print(m)

c = lin_reg.intercept_
print(c)

linear_model_pred = lin_reg.predict([[6.5]])
linear_model_pred

# polynomial model  ( bydefeaut degree - 2)
# polynomial regression (non linear model now)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
lin_reg_2 = LinearRegression()

lin_reg_2.fit(x_poly, y)

# poly nomial visualization

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('polymodel (Polynomial Regression)')
plt.xlabel('Possition Level')
plt.ylabel('Salary')
plt.show()

# prediction

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

# SUPPORT VECTOR REGRESSION MODEL

from sklearn.svm import SVR
svr_reg = SVR(kernel="poly",degree=4,gamma="auto",C = 10.0)
svr_reg.fit(x, y)

svr_model_pred = svr_reg.predict([[6.5]])
svr_model_pred

# K- NEAREST NEIGHBOUR

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=4,weights='uniform')
knn_reg.fit(x, y)

knn_model_pred = knn_reg.predict([[6.5]])
knn_model_pred

# DECISION TREE REGRESSOR

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='absolute_error')
dt_reg.fit(x, y)

dt_model_pred = dt_reg.predict([[6.5]])
dt_model_pred

# RANDOM FOREST 

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0)
rf_reg.fit(x, y)

rf_model_pred = rf_reg.predict([[6.5]])
rf_model_pred
