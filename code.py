# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 20:05:44 2018

@author: Abhi
"""

import pandas as pd
import seaborn as sns
import re

#Reading dataset from the csv file
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
dataset = [train, test]

#--------- Feature Engineering and Data Cleaning----------
#Pclass
print(train[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean())

#Sex
print(train[['Sex','Survived']].groupby(['Sex'], as_index = False).mean())

#SibSp and Parch can be combined to make a new feature called Family Size
for data in dataset:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean())

#Making another feature to see if the person is alone or not
for data in dataset:
    data['IsAlone'] = 0 #Giving all values of IsAlone as 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index = False).mean())    


#Filling Empty data
for data in dataset:  
    data['Age'].fillna(data['Age'].mean(), inplace = True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    data['Fare'].fillna(data['Fare'].median(), inplace = True)


#Getting Title from Names // NLP
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for data in dataset:
    data['Title'] = data['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))
print(pd.crosstab(test['Title'], test['Sex']))

#Cleaning up the Titles
for data in dataset:
    data['Title'] = data['Title'].replace(['Capt', 'Col', 'Countess', 'Lady', 'Dr', 'Jonkheer',
        'Major', 'Rev', 'Don', 'Sir', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')

print(train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean())

for data in dataset:
    data.drop('Name', axis = 1, inplace = True)
    data.drop('SibSp', axis = 1, inplace = True)
    data.drop('Parch', axis = 1, inplace = True)
    data.drop('Ticket', axis = 1, inplace = True)
    data.drop('Cabin', axis = 1, inplace = True)
    data.drop('PassengerId', axis = 1, inplace = True)

#------------------ Data Cleaning Ends -----------------

#------------------ Data Visualization & EDA ----------------

#Relation between FamilySize and Survival Percent divided by Sex
sns.pointplot(x = 'FamilySize', y = 'Survived', hue = 'Sex', data = train, palette = {'male':'orange', 'female':'purple'})

#Relation between Embarked, PClass and IsAlone vs Survived
sns.barplot(x = 'Embarked', y = 'Survived', hue = 'Sex', data = train)
sns.barplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = train)
sns.barplot(x = 'IsAlone', y = 'Survived', hue = 'Sex', data = train)

#How Pclass and Age effecting Survivability
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = train, split = True)

# How embarked factor with Pclass, Survival and Sex
e = sns.FacetGrid(train, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0)
e.add_legend()

#Heatmap for the dataset to find effect of one feature over other
fc = train.corr()
sns.heatmap(fc)

#------------------ Data Visualization Ends --------------

#---------------- Modeling of Data------------------

#Encoding Categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for data in dataset:
    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])
    data['Title'] = le.fit_transform(data['Title'])

#Taking X and y

X = train.iloc[:, 1:].values
y = train.iloc[:, 0].values

X_test = test.iloc[:, :].values


#OneHotEncoder
ohe = OneHotEncoder(categorical_features = [4])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:] #to avoid dummy trap
X_test = ohe.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

ohe2 = OneHotEncoder(categorical_features = [8])
X = ohe2.fit_transform(X).toarray()
X = X[ :, 1:]
X_test = ohe2.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.fit_transform(X_test)

#Fitting the SVM model
from sklearn.svm import SVC
classifier = SVC(C = 75, kernel = 'rbf', gamma = 0.02, random_state = 0)
classifier.fit(X, y) 
classifier.score(X, y)

"""
#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {'C' : [75, 100, 150], 'kernel' : ['rbf'], 'gamma': [0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_
"""

#Predicting the Test set
y_test = classifier.predict(X_test) #79.9% accuracy
#abc_test = abc.predict(X_test) #78.9%  accuracy

#Saving the predicted output in a csv file
y_out = pd.DataFrame(y_test) #Converting ndarray to dataframe
results = pd.DataFrame({'PassengerID':pd.read_csv('test.csv').iloc[:, 0], 'Survived':y_out.values[:, 0]})
results.to_csv('Submission.csv', header = True, index = False)


