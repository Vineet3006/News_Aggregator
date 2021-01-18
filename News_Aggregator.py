import re
import pickle
import numpy as np
import pandas as pd
from time import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Data

df = pd.read_csv("uci-news-aggregator.csv")

df.head(5)

df.columns

df[['TITLE', 'CATEGORY']]

print(len(df["TITLE"]))

#Pre processing

ps = PorterStemmer()
features = []

for i in range(0, len(df["TITLE"])):
    
    title = re.sub('[^a-zA-Z]', ' ', df["TITLE"][i])
    title = title.lower()
    title = title.split()
    
    title = [ps.stem(word) for word in title if not word in stopwords.words('english')]
    title = ' '.join(title)
    features.append(title)
    

#features

features[3]

pickle.dump( features, open("features.pkl", "wb") )


#Target variable

encoder = LabelEncoder()

labels = encoder.fit_transform(df['CATEGORY'])

print (labels)

#Test Train split

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state = 0, shuffle = True)

#TF-IDF Vectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test)

feature_names = vectorizer.get_feature_names()

print (features_train.shape)

print (features_test.shape)

#Feature selection

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train, labels_train)

features_train_transformed = selector.transform(features_train)
features_test_transformed  = selector.transform(features_test)

features_train_transformed.shape

print ("No of features after selection :", features_train_transformed.shape[1])

#Using MultinomialNB 

clf = MultinomialNB()

grid_param =     {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1,10]
    }

grid_search = GridSearchCV(estimator=clf, param_grid=grid_param, cv=5,
                           n_jobs =-1)

grid_search.fit(features_train_transformed, labels_train)

best_parameters = grid_search.best_params_

best_parameters

grid_search.best_score_

#Model creation

clf = MultinomialNB(alpha = 0.01)

model = clf.fit(features_train_transformed, labels_train)

clf.score(features_test_transformed,labels_test)

#Confusion matrix

predictions = clf.predict(features_test_transformed)

print (accuracy_score(labels_test, predictions))
print (confusion_matrix(labels_test, predictions))

#Predictions

pred = pd.DataFrame(predictions)

pred.to_csv("NB_predictions.csv", header=False, index=False)

#Save model

filename = 'MultinomialNB_model.pickle'

pickle.dump(model, open(filename, 'wb'))



#RandomForestClassifier

clf = RandomForestClassifier()

#

grid_param = {
         'criterion' : ('gini', 'entropy'),
         'n_estimators': [10, 50, 100, 150, 200],
         'max_features' : ('sqrt', 'log2'),
    }

grid_search = GridSearchCV(estimator=clf, param_grid=grid_param, cv=5,
                           n_jobs =-1)

grid_search.fit(features_train_transformed, labels_train)

best_parameters = grid_search.best_params_

best_parameters

grid_search.best_score_

#New model creation

clf = RandomForestClassifier(criterion = 'gini', n_estimators = 100, max_features = 'sqrt')

new_model = clf.fit(features_train_transformed, labels_train)

clf.score(features_test_transformed,labels_test)

#Confusion matrix

predictions = clf.predict(features_test_transformed)

print (accuracy_score(labels_test, predictions))
print (confusion_matrix(labels_test, predictions))


#Predictions

pred = pd.DataFrame(predictions)

pred.to_csv("RF_predictions.csv", header=False, index=False)

#Save model

filename = 'RandomForestClassifier_model.pickle'

pickle.dump(new_model, open(filename, 'wb'))
