import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


#Create Title

st.write(""" 
# Diabetes Detection
Detect if someone has diabetes using machine learning""")

#Open and display image in webapp

image=Image.open('Picture/ml.png')
st.image(image,caption="ML",use_column_width=True)

#Get Data
df = pd.read_csv('diabetes.csv')
st.subheader('Data Information')

#show data as table
st.dataframe(df)
#show statistics
st.write(df.describe())

#show data as chart

chart = st.bar_chart(df)

#Split the data into independent X and dependent Y variables
X=df.iloc[:,0:8].values
Y=df.iloc[:,-1].values

#Split data into training and testing set

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#function to get user input

def get_user_input():
    pregnancies=st.sidebar.slider('pregnancies',0,17,3) #min max default
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)  # min max default
    BloodPressure = st.sidebar.slider('BloodPressure', 50, 200, 80)  # min max default
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 23)  # min max default
    Insulin = st.sidebar.slider('Insulin', 0, 800, 30)  # min max default
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 20.4)  # min max default
    DiabetesPedigreeFunction= st.sidebar.slider('DiabetesPedigreeFunction', 0.07, 2.42, 0.4)  # min max default
    Age = st.sidebar.slider('Age', 21, 100, 30)  # min max default

    #storing in dictionary
    user={
        'pregnancies': pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    features=pd.DataFrame(user,index=[0])
    return features


#Get Input
userinput = get_user_input()

#Set a subheader and display the user input
st.subheader("User Input:")
st.write(userinput)

#Create and Train the Model
rfc=RandomForestClassifier()
rfc.fit(X_train, Y_train)

#Show the model metrics

st.subheader("Model Test Accuracy Score: ")

st.write(str(accuracy_score(Y_test, rfc.predict(X_test))*100)+"%")

#Store the models prediction in a variable
prediction=rfc.predict(userinput)

#Set Subheader and display classification

st.subheader("Classified as: ")

st.write(prediction)








