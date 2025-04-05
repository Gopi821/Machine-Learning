import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\Gopi Reddy\NIT7PM\mar/Data (1).csv')

# Independent variable
x = dataset.iloc[:, :-1].values
# Dependent variable
y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent') # mean,median

imputer = imputer.fit(x[:,1:3])

x[:, 1:3] = imputer.transform(x[:,1:3])

# IMPUTE CATEGORICAL VALUE FOR INDEPENDENT

from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0])

x[:,0]=labelencoder_x.fit_transform(x[:,0])

# Impute
labelencoder_y = LabelEncoder() 

y = labelencoder_y.fit_transform(x[:,0])


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,train_size=0.8,random_state=0)

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.3,random_state=0)
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.75,random_state=0)
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.9,random_state=0)

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.8,random_state=0)

# if we dont use the random state the records will change at time of running 
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.8)