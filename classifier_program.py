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

metadata = pd.read_csv(Path.cwd() / "Data" / "MetObjWithTags2.csv", encoding="utf-8")
metadata_df = pd.DataFrame(metadata)
#print("Metadata:")
#print(metadata.columns)
#print(metadata.head())

image_data = pd.read_csv(Path.cwd() / "Data" / "ImageColoursPivot.csv", encoding="utf-8")
image_data_df = pd.DataFrame(image_data)
print("Image Data:")
#print(image_data.columns)


metadata_df["Object ID"] = metadata_df["Object ID"].astype(str)
image_data_df["Object ID"] = image_data_df["Object ID"].astype(str).str.replace(".jpg","", regex=False)
print(image_data_df.head())
# Merge the two DataFrames on 'Object ID'
merged_df = pd.merge(image_data_df, metadata_df, how="left", left_on="Object ID", right_on="Object ID").fillna(0)
"""
# Check which columns contain Unicode characters
for col in merged_df.columns:
    try:
        str(merged_df[col].iloc[0]).encode('charmap')
    except (UnicodeEncodeError, IndexError):
        print(f"Unicode characters found in column: {col}")
"""
#print(merged_df.head())
print(merged_df.dtypes)

merged_df["Century"] = merged_df["Object Begin Date"].apply(lambda x: "21st Century" if x >= 2000 else "20th Century" if x >= 1900 else "19th Century" if x >= 1800 else "18th Century" if x >= 1700 else "17th Century" if x >= 1600 else "16th Century" if x >= 1500 else "15th Century" if x >= 1400 else "14th Century" if x >= 1300 else "13th Century" if x >= 1200 else "12th Century" if x >= 1100 else "11th Century" if x >= 1000 else "10th Century" if x >= 900 else "9th Century" if x >= 800 else "8th Century" if x >= 700 else "7th Century" if x >= 600 else "6th Century" if x >= 500 else "5th Century" if x >= 400 else "4th Century" if x >= 300 else "3rd Century" if x >= 200 else "2nd Century" if x >= 100 else "1st Century" if x >= 0 else "BC" if x < 0 else "Unknown")
merged_df["Century_short"] = merged_df["Object Begin Date"].apply(lambda x: "21st or 20th Century" if x >= 1900 else "19th Century" if x >= 1800 else "18th Century" if x >= 1700 else "17th Century" if x >= 1600 else "16th Century or 15th Century" if x >= 1400 else "Before 15th Century" if x < 1400 else "Unknown")
merged_df["Century_binary"] = merged_df["Century_short"].apply(lambda x: "18th century or later" if x in ["18th Century", "19th Century", "20th Century", "21st Century"] else "Before 18th century" if x in ["17th Century", "16th Century or 15th Century", "Before 15th Century"] else "Unknown")

merged_df = merged_df.loc[:, (merged_df != 0).any(axis=0)]



print(merged_df.columns)

merged_df.drop(columns=["Object Number", "Gallery Number"], inplace=True)

merged_df.to_csv(Path.cwd() / "Data" / "MetObjWithImageColoursMetadata.csv", index=False)

for c in merged_df.columns:
    counts = merged_df[c].value_counts()
    print("----")
    print(counts)