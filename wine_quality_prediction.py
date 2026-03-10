#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#load dataset
data = pd.read_csv("/kaggle/input/datasets/organizations/uciml/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


#data cleaning
data['quality']=data['quality'].apply(lambda x: 1 if x>5 else 0)

#split input and output
X=data.drop("quality",axis=1)
y=data["quality"]

#trian test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#logisticregression
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)
print("Logistic Regression Accuracy:",accuracy_score(y_test,lr_pred))

#decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

#random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))



# Corrected import for RandomForestClassifier and its usage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create model (using the correct class name)
rf_plot = RandomForestClassifier(random_state=42)

# Train
rf_plot.fit(X_train, y_train)

# Predict
dt_pred = rf_plot.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, dt_pred)
print("Random Forest Accuracy:", accuracy)


cm = confusion_matrix(y_test, dt_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Random Forest Accuracy")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


from sklearn.tree import plot_tree

plt.figure(figsize=(15,8))
plot_tree(dt, filled=True, max_depth=3, feature_names=X.columns)
plt.title("Random Forest Accuracy (Depth=3)")
plt.show()
