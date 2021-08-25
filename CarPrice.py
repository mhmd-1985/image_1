import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("CarPrice.csv")
data.head()
print(data.isnull().sum())
data.info()
sns.set_style('darkgrid')
sns.distplot(data.price)
plt.show()
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.show()
d=data.select_dtypes(['float64','int64'])
d.columns
s=data.select_dtypes(exclude='object')
s.columns
# X=np.array(data.drop([price],axis=1))

# y=np.array(data.price)
# X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.2)

# model = DecisionTreeRegressor()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
              "carwidth", "carheight", "curbweight", 
              "enginesize", "boreratio", "stroke", 
              "compressionratio", "horsepower", "peakrpm", 
              "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
