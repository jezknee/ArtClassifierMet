import pandas as pd
from pathlib import Path

meta_path = Path.cwd() / "Data" / "MetObjects.txt"
metadata = pd.read_csv(meta_path, encoding='utf-8')
df = pd.DataFrame(metadata)
print(df.head())

def get_tags(row):
    tags = dict() 
    if pd.notna(row["Tags"]):
        row_tags = row["Tags"].split("|")
        for r in row_tags:
            r = r.strip()
            tags[r] = tags.get(r, 0) + 1
    else:
        tags["No Tags"] = 1
    return tags


tags_df = pd.DataFrame(columns=["Object ID", "Tags"])
tags_data = []
for index, row in df.iterrows():
    image_tags = get_tags(row)
    #print(f"Image ID: {row['Object ID']}, Tags: {image_tags}")
    for tag, count in image_tags.items():
        tags_data.append({"Object ID": row['Object ID'], "Tags": tag})
tags_df = pd.DataFrame(tags_data)
tags_df = tags_df[tags_df["Tags"] != "No Tags"]
print(tags_df.head())