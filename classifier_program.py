from pathlib import Path 
import pandas as pd

pd.set_option("display.max_columns", None)

metadata = pd.read_csv(Path.cwd() / "Data" / "MetObjWithTags2.csv", encoding="utf-8")
metadata_df = pd.DataFrame(metadata)
print("Metadata:")
print(metadata.columns)
print(metadata.head())

image_data = pd.read_csv(Path.cwd() / "Data" / "ImageColoursPivot.csv", encoding="utf-8")
image_data_df = pd.DataFrame(image_data)
print("Image Data:")
print(image_data.columns)
print(image_data.head())

metadata_df["Object ID"] = metadata_df["Object ID"].astype(str)
image_data_df["Object ID"] = image_data_df["Object ID"].astype(str)

# Merge the two DataFrames on 'Object ID'
merged_df = pd.merge(metadata_df, image_data_df, how="left", left_on="Object ID", right_on="Object ID")
"""
# Check which columns contain Unicode characters
for col in merged_df.columns:
    try:
        str(merged_df[col].iloc[0]).encode('charmap')
    except (UnicodeEncodeError, IndexError):
        print(f"Unicode characters found in column: {col}")
"""
print(merged_df.head())
#a = merged_df.head().to_string().encode('utf-8', errors='replace').decode('utf-8')
