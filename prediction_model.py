import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics,neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pickle



dimensions_df = pd.read_csv('final_dataset.csv')


# print(dimensions_df.columns)
feature_df = dimensions_df[['Height', 'weight']]
# print(feature_df.columns)


#Independent variable
X = np.asarray(feature_df) 

#Dependent variable
y = np.asarray(dimensions_df['class'])


#split training and testing data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=123)


#######   kNN Classifier   #######


# classifier1 = KNeighborsClassifier(n_neighbors=1)
# classifier1.fit(X_train,y_train)
# y_predict1 = classifier1.predict(X_test)
# print("\nKNeighnorsClassifier - \n")
# print("accuracy: ",metrics.accuracy_score(y_test,y_predict1))
# print("precision: ",metrics.precision_score(y_test,y_predict1,average='weighted'))
# print("recall",metrics.recall_score(y_test,y_predict1,average='weighted'))
# print("f1",metrics.f1_score(y_test,y_predict1,average='weighted'))
# # print(acc2 )



# #######   SVM Classifier   #######

# classifier2 = svm.SVC(kernel='linear', gamma = 'auto', C=3)
# classifier2.fit(X_train,y_train)
# y_predict2 = classifier2.predict(X_test)
# print("accuracy: ",metrics.accuracy_score(y_test,y_predict2))
# print("precision: ",metrics.precision_score(y_test,y_predict2,average='weighted'))
# print("recall",metrics.recall_score(y_test,y_predict2,average='weighted'))
# print("f1",metrics.f1_score(y_test,y_predict2,average='weighted'))


# #######   Random Forest Classifier   #######

classifier3 = RandomForestClassifier(n_estimators=30)
classifier3.fit(X_train,y_train)
y_predict3 = classifier3.predict(X_test)
# print("accuracy: ",metrics.accuracy_score(y_test,y_predict3))
# print("precision: ",metrics.precision_score(y_test,y_predict3,average='weighted'))
# print("recall",metrics.recall_score(y_test,y_predict3,average='weighted'))
print(metrics.classification_report(y_test,y_predict3))


# #######   Naive Bayes Classifier   #######

# classifier4 = MultinomialNB()
# classifier4.fit(X_train,y_train)
# y_predict4 = classifier4.predict(X_test)
# print("accuracy: ",metrics.accuracy_score(y_test,y_predict4))
# print("precision: ",metrics.precision_score(y_test,y_predict4,average='weighted'))
# print("recall",metrics.recall_score(y_test,y_predict4,average='weighted'))
# print("f1",metrics.f1_score(y_test,y_predict4,average='weighted'))


# classifier4 = BernoulliNB(binarize=0.2)
# classifier4.fit(X_train,y_train)
# y_predict4 = classifier4.predict(X_test)
# print("accuracy: ",metrics.accuracy_score(y_test,y_predict4))
# print("precision: ",metrics.precision_score(y_test,y_predict4,average='weighted'))
# print("recall",metrics.recall_score(y_test,y_predict4,average='weighted'))
# print("f1",metrics.f1_score(y_test,y_predict4,average='weighted'))


# classifier4 = GaussianNB()
# classifier4.fit(X_train,y_train)
# y_predict4 = classifier4.predict(X_test)
# print("accuracy: ",metrics.accuracy_score(y_test,y_predict4))
# print("precision: ",metrics.precision_score(y_test,y_predict4,average='weighted'))
# print("recall",metrics.recall_score(y_test,y_predict4,average='weighted'))
# print("f1",metrics.f1_score(y_test,y_predict4,average='weighted'))



pickle.dump(classifier3,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


