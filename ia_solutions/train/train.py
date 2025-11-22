
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')


X=data.iloc[:,:-1]
y=data["fetal_health"]

scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=list(X.columns))

X_df.head()

X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

from sklearn.metrics import accuracy_score

"""# **Random Forest**"""

from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=10,
                                  random_state=42)
 
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=y_pred))

"""# **Modelos Ensemble**"""

from sklearn.ensemble import GradientBoostingClassifier


grd_clf = GradientBoostingClassifier(max_depth=10,
                                     n_estimators=100,
                                     learning_rate=0.01,
                                     random_state=42)
 
grd_clf.fit(X_train, y_train)

y_pred = grd_clf.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=y_pred))

