# Written by Finn Alessandrino 144941
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Draw histogram
def draw_hist (df):
    df.hist(grid = False, figsize=(10,11))

# Draw confusion matrix
def draw_conf_matrix (cf, name): 
    ax = sns.heatmap(cf, annot=True, fmt = '.2f')
    ax.set_title('Confusion Matrix ' + name)
    return ax

# Draw correlation matrix
def draw_corr_matrix (df):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.drop(df.columns[-1],axis=1).corr(), annot = True, 
                fmt = '.2f')

# Draw decision regions
def draw_dec_region (x, y, clf, X, clf_name):
    plt.figure()
    plot_decision_regions(x, y, clf, legend=2)
    plt.xlabel(X.columns[0], size=12)
    plt.ylabel(X.columns[1], size=12)
    plt.title(clf_name + ' Decision Region Boundary', size=14)

# SVM classifier
def classifier(X_train, y_train, X_test, classifier): 
    if (classifier == SVC):
        clf = classifier(kernel='rbf')
    else: 
        clf = classifier(n_neighbors=round(len(X_train)**0.5))
    
    clf.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    y_pred = clf.predict(X_test)
    return X_combined, y_combined, y_pred, clf

# Load data
wine_df = pd.read_csv('winequality.csv',header=0)

# Describe initial data
data_desc = wine_df.describe().round(2)
print(data_desc)

# Correlation between 'quality' and all other features
corr = pd.DataFrame(wine_df.corrwith(wine_df['quality']).
                    sort_values(ascending=False).drop('quality'))

# Preprocessing
wine_df["Wine Quality"] = [1 if x >= 7 else 0 for x in wine_df['quality']]
X = wine_df[[corr[0].idxmax(), corr[0].idxmin()]]
y = wine_df.iloc[:, -1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)
# Feature scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
X_combined, y_combined, y_pred, clf = classifier(X_train, y_train, X_test, SVC)
X_combined_knn, y_combined_knn, y_pred_knn, clf_knn = classifier(X_train, 
                                                                 y_train, 
                                                                 X_test, 
                                                                 KNeighborsClassifier)
# Model accuracy
cf_svm=metrics.confusion_matrix(y_test, y_pred)
cf_knn=metrics.confusion_matrix(y_test, y_pred)
print("Classification report SVM: ")
print(metrics.classification_report(y_test, y_pred))
print("Accuracy (this test): ", "{:.2f}".
      format((metrics.accuracy_score(y_test, y_pred))*100),"%")
print("KNN Classification report:")
print(metrics.classification_report(y_test, y_pred_knn))
print("Accuracy (this test): ", "{:.2f}".
      format((metrics.accuracy_score(y_test, y_pred_knn))*100),"%")

# Data visualization
draw_conf_matrix(cf_svm, "SVM")
# draw_conf_matrix(cf_knn, "KNN")
draw_corr_matrix((wine_df.drop(['Wine Quality'], axis=1)))
draw_hist((wine_df.drop(['Wine Quality'], axis=1)))
draw_dec_region(X_combined, y_combined, clf, X, "SVM")
draw_dec_region(X_combined_knn, y_combined_knn, clf_knn, X, "KNN")