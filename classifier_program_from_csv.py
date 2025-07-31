from pathlib import Path 
import pandas as pd
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pandas import read_csv 
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC

pd.set_option("display.max_columns", None)

merged = pd.read_csv(Path.cwd() / "Data" / "MetObjWithImageColoursMetadata.csv", encoding="utf-8")
merged_df = pd.DataFrame(merged)

# Prepare data for classification
print("Preparing data for classification...")

merged_df.drop(columns=["Department", "Object Name", "Title", "Object Begin Date", "Object End Date", "Medium", "Dimensions", "Century_binary", "Century_short"], inplace=True)
merged_df = merged_df.filter(regex='^Colour_', axis=1)
print(merged_df.head())
"""
for c in merged_df.columns:
    counts = merged_df[c].value_counts()
    print("----")
    print(counts)
"""
"""
names = merged_df.columns.tolist()
array = merged_df.values
X = array[:, 2:-1]  # Features (excluding 'Object ID' and 'Century')
Y = array[:, -1]    # Target variable (last column, 'Century')
test_size = 0.1
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
models = [] 
models.append(('LR', LogisticRegression(solver='liblinear'))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC())) 
# evaluate each model in turn 
results = [] 
names = [] 
classification_results = []
scoring = 'accuracy' 
for name, model in models: 
    kfold = KFold(n_splits=10, random_state=7,shuffle=True) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg) 

    model.fit(X_train, Y_train) 
    predicted = model.predict(X_test) 
    report = classification_report(Y_test, predicted, zero_division=0)
    classification_results.append([name, report])


# boxplot algorithm comparison 
fig = pyplot.figure()  
fig.suptitle('Algorithm Comparison')  
ax = fig.add_subplot(111)  
pyplot.boxplot(results)  
ax.set_xticklabels(names)  
pyplot.show()


print("Classification Reports:")
for model_name, rpt in classification_results:
    print(f"Model: {model_name}")
    print(rpt)
    print("="*50)
"""