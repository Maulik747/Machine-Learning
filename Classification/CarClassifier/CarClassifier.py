import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd

le=preprocessing.LabelEncoder()
data=pd.read_csv("car.data")
car_data=data.values
buying=le.fit_transform(list(data["buying"]))
maint=le.fit_transform(list(data["maint"]))
doors=le.fit_transform(list(data["doors"]))
persons=le.fit_transform(list(data["persons"]))
lug_boot=le.fit_transform(list(data["lug_boot"]))
safety=le.fit_transform(list(data["safety"]))
cls=le.fit_transform(list(data["class"]))

#attributes of the car we want to train
X=list(zip(buying,doors,persons,lug_boot,safety))

#the target attribute-class
y=list(cls)
X_train, X_test, y_train, y_test= sklearn.model_selection.train_test_split(X,y,test_size=0.1)

#using Kneighbour classifier, here K=5
model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print(le.inverse_transform(prediction))



