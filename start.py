from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
#pd.set_option("low_memory",False)

all_path = Path.cwd() / "Data" / "MetObjects.txt"
#print(path)
all_df = pd.read_csv(all_path, encoding="utf-8")
#print(all_df.head())
#print(all_df.columns)
#df.to_csv(Path.cwd() / "Data" / "MetObj.csv", index=False)

download_path = Path.cwd() / "Data" / "metadata.csv"
download_df = pd.read_csv(download_path, encoding="utf-8")

#print(download_df.columns)
#print(download_df.head())

new_df = pd.DataFrame(download_df[["object_id", "title", "department", "object_name", "medium", "classification", "artist_display_name"]])
#print(new_df.head())

merged_df = new_df.merge(how="left", right=all_df, left_on="object_id", right_on="Object ID")
merged_df = pd.DataFrame(merged_df[["object_id", "title", "department", "object_name", "medium", "artist_display_name", "Object Date", "Object Begin Date", "Object End Date", "Tags"]])
print(merged_df.head())

merged_df["Century_short"] = merged_df["Object Begin Date"].apply(lambda x: "21st or 20th Century" if x >= 1900 else "19th Century" if x >= 1800 else "18th Century" if x >= 1700 else "17th Century" if x >= 1600 else "16th Century or 15th Century" if x >= 1400 else "Before 15th Century" if x < 1400 else "Unknown")
merged_df["Century_binary"] = merged_df["Century_short"].apply(lambda x: "18th century or later" if x in ["18th Century", "19th Century", "20th Century", "21st Century"] else "Before 18th century" if x in ["17th Century", "16th Century or 15th Century", "Before 15th Century"] else "Unknown")
merged_df["Century"] = merged_df["Object Begin Date"].apply(lambda x: "21st Century" if x >= 2000 else "20th Century" if x >= 1900 else "19th Century" if x >= 1800 else "18th Century" if x >= 1700 else "17th Century" if x >= 1600 else "16th Century" if x >= 1500 else "15th Century" if x >= 1400 else "14th Century" if x >= 1300 else "13th Century" if x >= 1200 else "12th Century" if x >= 1100 else "11th Century" if x >= 1000 else "10th Century" if x >= 900 else "9th Century" if x >= 800 else "8th Century" if x >= 700 else "7th Century" if x >= 600 else "6th Century" if x >= 500 else "5th Century" if x >= 400 else "4th Century" if x >= 300 else "3rd Century" if x >= 200 else "2nd Century" if x >= 100 else "1st Century" if x >= 0 else "BC" if x < 0 else "Unknown")
for c in merged_df.columns:
    print("----")
    print(merged_df[c].value_counts())

merged_df.to_csv(Path.cwd() / "Data" / "MetObjFiltered.csv", index=False)