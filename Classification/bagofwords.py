import sklearn
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download('stopwords')
message4='hello hello world world play'
message5='row row row row column joker'

def process_text(text):
    #remove stopwords
    #remove punctuation
    #return a list of clear text
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc=''.join(nopunc)
    print(nopunc)
    print(nopunc.split()) # turn the words having no punctuation to a list of words seperated by comma

    clean_words=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean_words

bow4=CountVectorizer(analyzer=process_text).fit_transform([[message4],[message5]])
print(bow4)

