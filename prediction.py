import pandas as pd
import numpy as np
from nltk import RegexpTokenizer
import string
import re
from nltk.corpus import stopwords 
import nltk 
from nltk.stem.snowball import SnowballStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes, svm
from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier

#nltk.download('stopwords')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

import pickle
# Preprocessing
class Prediction:
    def __init__(self, text):
        self.text = text

    #"first stage preprocessing" steps
    def preprocessing(self):
        self.df = pd.read_csv('static/models/resampled_comments_1.csv')
        self.comments = self.df[['comment', 'rating', 'sentiment']]
        self.comments['comment'] = self.comments['comment'].map(lambda x: x.lower())
        
        toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
        token = self.comments.apply(lambda row: toknizer.tokenize(row['comment']),axis= 1)
        
        stop_words = set(stopwords.words('french'))
        stop_token = token.apply(lambda x: [item for item in x if item not in stop_words])
        
        stemmer = SnowballStemmer(language='french')
        stemm = stop_token.apply(lambda x: [stemmer.stem(y) for y in x])
        
        lemmatizer = FrenchLefffLemmatizer()
        lemm = stemm.apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
        
        for i in range(len(lemm)):
            lemm[i] = ' '.join(lemm[i])
        
        self.comments['lemmatiser_com'] = lemm
        data = self.comments[['comment','lemmatiser_com','sentiment']]
        
        self.df = pd.DataFrame(data)
        return self.df

    #create the vectorizer that will be used in production
    def getprediction(self, input, option):

        cv = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('french'))
        
        # Separate data and labels
        X = self.comments['comment']
        y = self.comments['sentiment']
        
        # Using a hashing vectorizer to keep model size low
        cv.fit(X)
        X_fitted = cv.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_fitted, y, test_size=0.25, random_state=1)

        # Linear SVM powered by SGD Classifier
        if(option == "svm"):
            print("*********** using svm *******************")
            clf = SGDClassifier(loss="hinge", tol=None, max_iter=10)
            clf.fit(X_train,y_train)
            clf.score(X_test,y_test)
            y_pred = clf.predict(X_test)

        # RandomForestClassifier
        elif(option == "rc"):
            print("*********** using RandomForestClassifier *******************")
            clf = RandomForestClassifier(n_estimators=1000, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        elif(option =="sc"):
            print("*********** using StackingClassifier *******************")

            # TF-IDF matrice
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            mnb = naive_bayes.MultinomialNB()
            mnb.fit(X_train.todense(), y_train)
            svc_model = SVC(gamma='auto')
            svc_model.fit(X_train, y_train)
            estimators = [('Random forest', rf), ("Naive bayes", mnb),("SVM", svc_model)]

            clf = StackingClassifier(estimators=estimators, final_estimator=SVC())
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)


        cf_matrix = confusion_matrix(y_test, y_pred)
        self.df_cm = pd.DataFrame(cf_matrix, range(3),range(3))

        #Classification Report
        self.report = classification_report(y_test, y_pred, output_dict=True)
        
        return clf.predict(cv.transform(input))

    def getsampledata(self, size):
        return self.df.head(size)

    def getsampledata2(self, size):
        df2 = pd.read_csv('static/models/resampled_comments_1.csv')
        df2 = df2[['rating','comment','bonus_info','city']]
        #df2.drop('unnamed:0', axis=1)
        return df2.head(size)

    def getcr(self):
        return pd.DataFrame(self.report).transpose()

    def create_cm(self):
        # plot (powered by seaborn)
        fig = Figure()
        plt.rcParams["figure.figsize"] = (15,10)
        ax = fig.add_subplot(1, 1, 1)
        
        sn.set(font_scale=1)
        sn.heatmap(self.df_cm, ax = ax, annot=True,annot_kws={"size": 16}, fmt='g')

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix') 
        ax.xaxis.set_ticklabels(['negative', 'positive', 'neutral'])
        ax.yaxis.set_ticklabels(['negative', 'positive', 'neutral'])
        
        return fig

