from pathlib import Path 
import pandas as pd
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn import svm
from numpy import set_printoptions, sqrt
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pandas import read_csv 
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 

pd.set_option("display.max_columns", None)



merged = pd.read_csv(Path.cwd() / "Data" / "MetObjWithImageColoursMetadata.csv", encoding="utf-8")
merged_df = pd.DataFrame(merged)
print(merged_df.columns)

# Prepare data for classification
print("Preparing data for classification...")

merged_df.drop(columns=["Department", "Object Name", "Title", "Object End Date", "Medium", "Dimensions", "Century", "Century_binary", "Century_short"], inplace=True)
#merged_df = merged_df.filter(regex='^(Object ID|Colour_|Century)', axis=1)
print(merged_df.columns)
"""
for c in merged_df.columns:
    counts = merged_df[c].value_counts()
    print("----")
    print(counts)
"""
"""
century_counts = merged_df["Century"].value_counts()
century_list = []
for century, count in century_counts.items():
    if count >= 100:
        century_list.append(century)

merged_df = merged_df[merged_df["Century"].isin(century_list)]
print(merged_df.columns )

century_counts = merged_df["Century"].value_counts()
print("Century Counts:")
print(century_counts)
"""
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
print(merged_df.columns)
# Now you can drop the individual color columns and keep only the groups
object_id_col = merged_df[["Object ID"]]
centur_col = merged_df[["Object Begin Date"]]
temp_df = merged_df = merged_df.drop(columns=["Object ID", "Object Begin Date"])
group_columns = [col for col in temp_df.columns if col.startswith('ColorGroup_')]
other_columns = [col for col in temp_df.columns if not col.startswith('Colo')]
df = pd.concat([object_id_col, temp_df[other_columns], temp_df[group_columns], centur_col], axis=1)
print(df.columns)

names = df.columns.tolist()
array = df.values
X = array[:, 2:-1]  # Features (excluding 'Object ID' and 'Object Begin Date')
Y = array[:, -1]    # Target variable (last column, 'Object Begin Date')
test_size = 0.1
seed = 7

# feature extraction 
test = SelectKBest(score_func=f_regression, k=25) 
fit = test.fit(X, Y) 
# summarize scores 
set_printoptions(precision=3) 
print(fit.scores_) 
features = fit.transform(X) 
# summarize selected features 
print(features[0:25,:]) 

feature_names = names[2:-1]  # This matches your X array slice

# Get which features were selected
selected_mask = test.get_support()

# Get the names of selected features
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
feature_list = [feature for feature in selected_features]
print("Selected Features:")
for i, feature in enumerate(selected_features):
    print(f"{i+1}. {feature}")

df2 = df.filter(items=["Object ID"] + selected_features + ["Object Begin Date"], axis=1)
print(df2.head())
print(df2.columns)


names = df2.columns.tolist()
array = df2.values
X = array[:, 2:-1]  # Features (excluding 'Object ID' and 'Object Begin Date')
Y = array[:, -1]    # Target variable (last column, 'Object Begin Date')
test_size = 0.1
seed = 7


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
models = [] 
models.append(('SVR', svm.SVR())) 
models.append(('RFR', RandomForestRegressor())) 
models.append(('LnR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DTR', DecisionTreeRegressor()))
# evaluate each model in turn 
results = [] 
model_names = [] 
regression_results = []
scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

for name, model in models: 
    kfold = KFold(n_splits=10, random_state=7,shuffle=True) 
    mse_scores = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
    mae_scores = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_error') 
    r2_scores = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='r2')
    
    results.append(r2_scores)  # Use R² for the boxplot
    model_names.append(name)
    
    # Print results
    print(f"{name}:")
    print(f"  RMSE: {sqrt(-mse_scores.mean()):.2f} (+/- {sqrt(mse_scores.std() * 2):.2f})")
    print(f"  MAE: {-mae_scores.mean():.2f} (+/- {mae_scores.std() * 2:.2f})")
    print(f"  R²: {r2_scores.mean():.3f} (+/- {r2_scores.std() * 2:.3f})")
    
    # Fit model and get test predictions
    model.fit(X_train, Y_train) 
    predicted = model.predict(X_test)
    
    # Calculate test metrics
    test_mse = mean_squared_error(Y_test, predicted)
    test_mae = mean_absolute_error(Y_test, predicted)
    test_r2 = r2_score(Y_test, predicted)

    
    # Create predictions DataFrame - need to track test indices
    # Get the original indices used in train_test_split
    X_train_with_ids, X_test_with_ids, Y_train, Y_test = train_test_split(
        df2[['Object ID'] + selected_features], df2['Object Begin Date'], 
        test_size=test_size, random_state=seed
    )

    # Get Object IDs for test set
    test_object_ids = X_test_with_ids['Object ID'].values

    predictions_df = pd.DataFrame({
        'Object ID': test_object_ids,
        'Predicted': predicted,
        'Actual': Y_test,
        'Error': predicted - Y_test
    })
    predictions_df.to_csv(Path.cwd() / "Data" / f"{name}_predictions.csv", index=False)
    
    regression_results.append({
        'model': name,
        'test_rmse': sqrt(test_mse),
        'test_mae': test_mae,
        'test_r2': test_r2
    })
    print(f"  Test RMSE: {sqrt(test_mse):.2f}")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test R²: {test_r2:.3f}")
    print("-" * 40)


# Boxplot algorithm comparison (using R² scores)
fig = pyplot.figure(figsize=(10, 6))
fig.suptitle('Regression Algorithm Comparison (R² Score)')  
ax = fig.add_subplot(111)  
pyplot.boxplot(results)  
ax.set_xticklabels(model_names)
ax.set_ylabel('R² Score')
pyplot.xticks(rotation=45)
pyplot.show()

print("\nRegression Results Summary:")
print("Model\t\tTest RMSE\tTest MAE\tTest R²")
print("-" * 50)
for result in regression_results:
    print(f"{result['model']}\t\t{result['test_rmse']:.2f}\t\t{result['test_mae']:.2f}\t\t{result['test_r2']:.3f}")