from numpy import unique
from numpy import where
import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from matplotlib import pyplot

stud_data= pd.read_csv("student-mat.csv", sep=";")
model=AgglomerativeClustering(n_clusters=2)
X1=stud_data['G1']
X2=stud_data['G2']
G1=list(X1.values)
G2=list(X2.values)
le=preprocessing.LabelEncoder()
G1=np.array(G1)
G2=np.array(G2)

G1=le.fit_transform(G1)
G2=le.fit_transform(G2)
Grades=list(zip(G1,G2))
Grades=np.asanyarray(Grades)
print(Grades)
yhat = model.fit_predict(Grades)
clusters = unique(yhat)
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(Grades[row_ix, 0], Grades[row_ix, 1])
# show the plot
pyplot.show()
