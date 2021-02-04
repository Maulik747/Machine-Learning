import numpy as np
import pandas as pd
import nltk
import sklearn
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv("smaller_email_spam2.csv")

#check for duplicates and remove them
df.drop_duplicates(inplace=True)
def process_text(text):
    #remove stopwords
    #remove punctuation
    #return a list of clear text
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc=''.join(nopunc)
    clean_words=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

#convert a collection of text to matrix of tokens
messages_bow=CountVectorizer(analyzer=process_text).fit_transform(df['EmailText'])

#split data into 80% training, 20% testing
X_train, X_test, y_train, y_test= sklearn.model_selection.train_test_split(messages_bow,df['Label'],test_size=0.2,random_state=0)


#naive bayes classifier
classifier=MultinomialNB().fit(X_train,y_train)

print(X_test)

#print the prediction done by the model
print(classifier.predict(X_test))








