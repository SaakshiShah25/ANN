#ANN

#Data Preprocessing
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X = X.astype('float64')
X=X[:,1: ]

#train set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test,copy=None)

#Fitting classifier to training set
#classifier here
#Part 2
#importing keras and libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

'''#Initialising the ANN
classifier=Sequential()

#Adding input layer and hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

#Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)



#Predicting the test set results
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
mean=accuracies.mean()
variance=accuracies.std()'''

#Tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={
            'batch_size':[25,32],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']
            }
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_
