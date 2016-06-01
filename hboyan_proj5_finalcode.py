#Prepare the environment
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

'''
Pre-Task: Describe the goals of your study

The goal of this study is to examine the data about survivors of the Titanic disaster in April 1912.
We will determine the most important features in predicting survival, and create the best model available
to predict survival for a given individual, based on their situational data.
'''

'''
Part 1: Aquire the Data
'''

# 1. Connect to the remote database
import psycopg2

#code sourced from https://wiki.postgresql.org/wiki/Using_psycopg2_with_PostgreSQL
def main(table):
    #Define our connection string
    conn_string = "host='dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com' port=5432 user='dsi_student' dbname='titanic' password='gastudents'"
    # print the connection string we will use to connect
    print "Connecting to database\n	->%s" % (conn_string)
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)
    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()
    print "Connected!\n"

    # execute our Query
    cursor.execute("SELECT * FROM "+table)

    # retrieve the records from the database
    records = cursor.fetchall()

    return (records)

# 2. Query the database and aggregate the data
data = main('train')

columns = ['index', 'passengerID', 'survived', 'class', 'name', 'sex', 'age', 'siblings', 'par_ch', 'ticket', 'fare', 'cabin', 'embarked']
df = pd.DataFrame(data, columns=columns)
df.head()


# 3. What are the risks and assumptions of our data?
# - Not all passengers are represented (there were more people than this)
# - Assuming accurate reporting
# - Data may be missing
# - Siblings/parents/children unclear - only counting on board or in total?

'''
Part 2: Exploratory Data Analysis
1. Describe the Data
2. Visualize the Data
'''
df.head()
df.describe()
df.isnull().sum()
df.dtypes
pd.tools.plotting.scatter_matrix(df)


'''
Part 3: Data Wrangling
1. Create Dummy Variables for Sex
'''
df = df.drop('cabin', axis=1)
df = df.drop('ticket', axis=1)
df = df.drop('index', axis=1)

df = df.dropna()

df['sex'] = df['sex'] == 'male'
df['sex'] = df['sex'].astype(int)

'''
Part 4: Logistic Regression and Model Validation
'''
# 1. Define the variables that we will use in our classification analysis
df.dtypes

cols = ['class','sex','age','siblings','par_ch']
# left out: passengerID (unique identifier), survived(target), name, fare (too much variance,
# really just a proxy for class), embarked (not relevant)
X = df[cols]

# 2. Transform "Y" into a 1-Dimensional Array for SciKit-Learn
y = list(df.survived)

# 3. Conduct the logistic regression
lr = LogisticRegression()
lr.fit(X,y)

# 4. Examine the coefficients to see our correlations
results = pd.DataFrame(zip(cols, (x for x in lr.coef_[0])))
results

# 6. Test the Model by introducing a Test or Validaton set
X_train, X_test, y_train, y_test = train_test_split(X,y)
lr_test = LogisticRegression()
lr_test.fit(X_train,y_train)

# 7. Predict the class labels for the Test set
y_pred = lr_test.predict(X_test)

# 8. Predict the class probabilities for the Test set
y_proba = lr_test.predict_proba(X_test)

# 9. Evaluate the Test set
lr_test.score(X_test,y_test)

# 10. Cross validate the test set
cv_test = cross_val_score(lr_test, X_test, y_test)
cv_test.mean()
cv_test.std()

# 11. Check the Classification Report
cr = classification_report(y_test, y_pred)
print cr

# 12. What do the classification metrics tell us?
# Precision: Ratio of correctly predicted true positivies to all predicted positives
# Recall: Ratio of correctly predicted true positives to all true positives in sample
# f1-score: Harmonic mean of precision and recall

# 13. Check the Confusion Matrix
cm = pd.DataFrame(confusion_matrix(y_test,y_pred), index=['true_dead','true_alive'], columns=['pred_dead','pred_alive'])
cm

# 14. What does the Confusion Matrix tell us?
# How well our predictions match reality

# 15. Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1])
plt.plot(fpr,tpr)
plt.xlabel('False Pos Rate (1 - Specificity)')
plt.ylabel('True Pos Rate (Sensitivity)')
plt.show()

# 16. What does the ROC curve tell us?
# It tells us how well this model achieves the balance between sensitivity and Specificity
# An ideal curve would be like an upside down L shape. The further the line is from x=1,
# the more area under the curve, the better the model.

'''
Part 5: Gridsearch
'''

# 1. Use GridSearchCV with logistic regression to search for optimal parameters
# Use the provided parameter grid. Feel free to add if you like (such as n_jobs).
# Use 5-fold cross-validation.

logreg_parameters = {
    'penalty':['l1','l2'],
    'C':np.logspace(-5,1,50),
    'solver':['liblinear']
}

gs = GridSearchCV(lr, logreg_parameters, cv=5)
gs.fit(X_train,y_train)

# 2. Print out the best parameters and best score. Are they better than the vanilla logistic regression?
gs.best_params_
gs.best_score_
# Yes, this is an improvement, albeit slight

# 3. Explain the difference between the difference between the L1 (Lasso) and L2 (Ridge)
# penalties on the model coefficients.
# Lasso estimates "sparse coefficients", so it tends to return a model with fewer parameters
# on which the result is dependent. Ridge uses linear least squares as its loss function,
# and has regularization built in.


# 4. What hypothetical situations are the Ridge and Lasso penalties useful?
# Lasso is great for a model with few parameters. Ridge is an all or nothing model, meaning it
# cannot zero coefficients, so it either uses all or none of the parameters. It's great for
# models involving multivariate targets.

'''
Part 6: Gridsearch and kNN
'''

# 1. Perform Gridsearch for the same classification problem as above, but use KNeighborsClassifier
# as your estimator. At least have number of neighbors and weights in your parameters dictionary.
 k_vals = range(1,100)
 weights = ['uniform', 'distance']
 knn_params = dict(n_neighbors=k_vals, weights=weights)

 knn = KNeighborsClassifier()
 gs_knn = GridSearchCV(knn, knn_params)
 gs_knn.fit(X_train,y_train)

# 2. Print the best parameters and score for the gridsearched kNN model.
# How does it compare to the logistic regression model?
print gs_knn.best_params_
print gs_knn.best_score_

# 3. How does the number of neighbors affect the bias-variance tradeoff of your model?
# Increasing k will decrease variance and increase bias.
# Decreasing k will increase variance and decrease bias.
# Source: http://scott.fortmann-roe.com/docs/BiasVariance.html

# 4. In what hypothetical scenario(s) might you prefer logistic regression over kNN, aside from
# model performance metrics?
# If you want to train, use logistic regression. If you want to tune, use KNN.
# If your decision boundary is linear, LR is great, but KNN can handle non-linear boundaries.
# If you want probabilities of confidence, use LR. KNN predicts only labels.
# If your training set is small, LR is better. KNN works better for larger training sets.
# https://www.quora.com/How-can-we-compare-the-advantages-and-disadvantage-of-logistic-regression-versus-k-nearest-neighbour
# http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/

# 5. Fit a new kNN model with the optimal parameters found in gridsearch.
knn_best = KNeighborsClassifier(n_neighbors=6, weights='distance')
knn_best.fit(X_train,y_train)
y_pred_knn = knn_best.predict(X_test)
y_probs_knn = knn_best.predict_proba(X_test)

# 6. Construct the confusion matrix for the optimal kNN model. Is it different from the
# logistic regression model? If so, how?
confusion_matrix(y_pred, y_test)
confusion_matrix(y_pred_knn, y_test)
# It gets more wrong, but is fairly similar.

# 7. [BONUS] Plot the ROC curves for the optimized logistic regression model and the optimized
# kNN model on the same plot.
fpr_knn, tpr_knn, thresholds = roc_curve(y_test, y_probs_knn[:,1])
plt.plot(fpr,tpr, color='blue')
plt.plot(fpr_knn,tpr_knn, color='red')
plt.xlabel('False Pos Rate (1 - Specificity)')
plt.ylabel('True Pos Rate (Sensitivity)')
plt.show()

'''
SUMMARY:
The logistic regression gave us a cross validated score of 0.77 (+/- 0.02). It had an f1 score of 0.8.
An optimised logistic regression using GridSearch was able to get the score up to 0.81.
KNN was a less effective model, only achieving a score of 0.775, even using gridsearch.
The ROC curves confirm this, with the logistic regression sitting comfortably above the KNN curve. 
'''
