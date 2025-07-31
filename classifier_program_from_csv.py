from pathlib import Path 
import pandas as pd
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from numpy import set_printoptions 
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 

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
merged_df = merged_df.filter(regex='^(Object ID|Colour_|Century)', axis=1)
#print(merged_df.head())
"""
for c in merged_df.columns:
    counts = merged_df[c].value_counts()
    print("----")
    print(counts)
"""
century_counts = merged_df["Century"].value_counts()
century_list = []
for century, count in century_counts.items():
    if count >= 100:
        century_list.append(century)

merged_df = merged_df[merged_df["Century"].isin(century_list)]

century_counts = merged_df["Century"].value_counts()
print("Century Counts:")
print(century_counts)

# Define color groups based on common web color names
color_groups = {
    'whites': ['white', 'snow', 'ivory', 'linen', 'beige', 'aliceblue', 'ghostwhite', 
               'whitesmoke', 'seashell', 'oldlace', 'floralwhite', 'lightcyan', 'honeydew'],
    'blacks': ['black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray', 'gainsboro'],
    'reds': ['red', 'darkred', 'crimson', 'firebrick', 'indianred', 'lightcoral', 
             'salmon', 'darksalmon', 'lightsalmon', 'pink', 'lightpink', 'hotpink'],
    'blues': ['blue', 'navy', 'darkblue', 'mediumblue', 'royalblue', 'steelblue', 
              'lightsteelblue', 'lightblue', 'powderblue', 'lightskyblue', 'skyblue'],
    'greens': ['green', 'darkgreen', 'forestgreen', 'limegreen', 'lime', 'lightgreen',
               'palegreen', 'darkseagreen', 'mediumseagreen', 'seagreen', 'olive'],
    'yellows': ['yellow', 'gold', 'orange', 'darkorange', 'lightyellow', 'lemonchiffon',
                'lightgoldenrodyellow', 'papayawhip', 'moccasin', 'peachpuff'],
    'purples': ['purple', 'indigo', 'darkviolet', 'blueviolet', 'mediumorchid', 
                'plum', 'violet', 'magenta', 'fuchsia', 'darkmagenta'],
    'browns': ['brown', 'maroon', 'darkgoldenrod', 'goldenrod', 'rosybrown', 
               'sandybrown', 'tan', 'burlywood', 'wheat', 'navajowhite', 'bisque']
}

# Function to map colors to groups
def get_color_group(color_name):
    color_name = color_name.lower()
    for group, colors in color_groups.items():
        if color_name in colors:
            return group
    return 'other'  # For colors not in any group

# Apply grouping to your DataFrame
# Get all Colour_ columns
colour_columns = [col for col in merged_df.columns if col.startswith('Colour_')]

# Create new grouped columns
for group_name in color_groups.keys():
    # Find all columns that belong to this color group
    group_columns = []
    for col in colour_columns:
        color_name = col.replace('Colour_', '')  # Extract color name
        if get_color_group(color_name) == group_name:
            group_columns.append(col)
    
    # Sum all columns in this group
    if group_columns:
        merged_df[f'ColorGroup_{group_name}'] = merged_df[group_columns].sum(axis=1)
        print(f"{group_name}: {len(group_columns)} colors grouped")

# Now you can drop the individual color columns and keep only the groups
group_columns = [col for col in merged_df.columns if col.startswith('ColorGroup_')]
other_columns = [col for col in merged_df.columns if not col.startswith('Colour_')]
df = merged_df[["Object ID"] + group_columns + ["Century"]]

names = df.columns.tolist()
array = df.values
X = array[:, 2:-1]  # Features (excluding 'Object ID' and 'Century')
Y = array[:, -1]    # Target variable (last column, 'Century')
test_size = 0.1
seed = 7
"""
# feature extraction 
test = SelectKBest(score_func=chi2, k=10) 
fit = test.fit(X, Y) 
# summarize scores 
set_printoptions(precision=3) 
print(fit.scores_) 
features = fit.transform(X) 
# summarize selected features 
print(features[0:10,:]) 

feature_names = names[2:-1]  # This matches your X array slice

# Get which features were selected
selected_mask = test.get_support()

# Get the names of selected features
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
feature_list = [feature for feature in selected_features]
print("Selected Features:")
for i, feature in enumerate(selected_features):
    print(f"{i+1}. {feature}")

df = df.filter(items=["Object ID"] + selected_features + ["Century"], axis=1)
print(df.head())
print(df.columns)
"""

names = df.columns.tolist()
array = df.values
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
