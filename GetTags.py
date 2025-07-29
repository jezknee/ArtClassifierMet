import pandas as pd
from pathlib import Path

meta_path = Path.cwd() / "Data" / "MetObjects.txt"
metadata = pd.read_csv(meta_path, encoding='utf-8')
df = pd.DataFrame(metadata)
#print(df.head())

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
#print(tags_df.head())
tags_unpivot = tags_df.pivot_table(index='Object ID', columns='Tags', aggfunc='size', fill_value=0)
tags_unpivot = tags_unpivot.reset_index()
print(tags_unpivot.head())

df2 = pd.merge(df, tags_unpivot, how="left", left_on="Object ID", right_on="Object ID")
print(df2.head())
#df2.to_csv(Path.cwd() / "Data" / "MetObjWithTags.csv", index=False)