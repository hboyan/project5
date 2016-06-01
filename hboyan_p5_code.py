# import pandas as pd
# import sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

#Part 1 - Acquire the Data
df = pd.read_csv('train.csv')

#Part 2 - EDA
df.head()
df.describe()
df.isnull().sum()

#sns.pairplot(df)

#Part 3 - Data Wrangling
df = df.drop('Cabin', axis=1)
df = df.dropna()

df['Sex'] = df['Sex'] == 'male'
df['Sex'] = df['Sex'].astype(int)


#Part 4 - Logistic Regression & Model Validation
df.columns

numcols = ['Pclass','Sex','Age','SibSp','Parch','Fare']
X=df[numcols]
y=df['Survived']

# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split, cross_val_score
lr = LogisticRegression()

lr.fit(X,y)
results = pd.DataFrame(zip(numcols, (x for x in lr.coef_[0])))
results

X_train, X_test, y_train, y_test = train_test_split(X,y)
lr.fit(X_train,y_train.values)
results2 = pd.DataFrame(zip(numcols, (x for x in lr.coef_[0])))
results2

results2cv = cross_val_score(lr, X_test, y_test)
results2cv.mean()

pd.value_counts(df['Survived'])
#Class labels: 0 = died, 1 = survived

y_pred = lr.predict(X_test)
y_pred
y_probs = lr.predict_proba(X_test)
y_probs
lr.score(X_test,y_test)
cross_val_score(lr, X_test,y_test).mean()

# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
cr = classification_report(y_test, y_pred)
print cr

cm = confusion_matrix(y_test,y_pred)
cm


# from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_probs[:,1])
plt.plot(fpr,tpr)
plt.xlabel('False Pos Rate (1 - Specificity)')
plt.ylabel('True Pos Rate (Sensitivity)')
plt.show()
#Part 5 - GridSearch
# from sklearn.grid_search import GridSearchCV
# import numpy as np

logreg_parameters = {'penalty':['l1','l2'],'C':np.logspace(-5,1,50),'solver':['liblinear']}

gs = GridSearchCV(lr, logreg_parameters, cv=5)
gs.fit(X_train,y_train)
gs.best_params_

gs.best_score_

#Part 6 - GridSearch and KNN
# from sklearn.neighbors import KNeighborsClassifier

k_vals = range(1,100)
weights = ['uniform', 'distance']
knn_params = dict(n_neighbors=k_vals, weights=weights)

knn = KNeighborsClassifier()
gs_knn = GridSearchCV(knn, knn_params)
gs_knn.fit(X_train,y_train)

gs_knn.best_params_
gs_knn.best_score_

knn_best = KNeighborsClassifier(n_neighbors=21, weights='distance')
knn_best.fit(X_train,y_train)
y_pred_knn = knn_best.predict(X_test)
y_probs_knn = knn_best.predict_proba(X_test)

confusion_matrix(y_pred, y_test)
confusion_matrix(y_pred_knn, y_test)

fpr_knn, tpr_knn, thresholds = roc_curve(y_test, y_probs_knn[:,1])
plt.plot(fpr,tpr, color='blue')
plt.plot(fpr_knn,tpr_knn, color='red')
plt.xlabel('False Pos Rate (1 - Specificity)')
plt.ylabel('True Pos Rate (Sensitivity)')
plt.show()

#Part 7 - Precision-recall

gs_lr2 = GridSearchCV(lr, logreg_parameters, cv=5, scoring='average_precision')
gs_lr2.fit(X_train,y_train)

gs.best_params_
gs.best_score_

gs_lr2.best_params_
gs_lr2.best_score_

y_pred_lr2 = gs_lr2.predict(X_test)
y_probs_lr2 = gs_lr2.predict_proba(X_test)

cm_lr2 = confusion_matrix(y_pred_lr2, y_test)
print cm_lr2

# from sklearn.metrics import precision_recall_curve

'''
The precision is the ratio tp / (tp + fp) where tp is the number of true positives and
fp the number of false positives. The precision is intuitively the ability of the classifier
not to label as positive a sample that is negative. (Avoid false positives)

The recall is the ratio tp / (tp + fn) where tp is the number of true positives and
fn the number of false negatives. The recall is intuitively the ability of the classifier
to find all the positive samples. (Avoid false negatives)
'''

prec_lr2, recall_lr2, thresholds = precision_recall_curve(y_test, y_probs_lr2[:,1])
# plt.plot(fpr,tpr, color='blue')
# plt.plot(fpr_knn,tpr_knn, color='red')
plt.plot(prec_lr2,recall_lr2, color='green')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()
