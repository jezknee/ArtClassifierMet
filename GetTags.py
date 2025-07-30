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
#print(tags_unpivot.head())

df2 = pd.merge(df, tags_unpivot, how="left", left_on="Object ID", right_on="Object ID")
df2 = df2.fillna(0)
print(df2.columns)
df2.drop(columns=['Is Highlight', 'Is Timeline Work', 'Is Public Domain','AccessionYear','Culture', 'Period', 'Dynasty', 'Reign', 'Portfolio', 'Constituent ID', 'Artist Role', 'Artist Prefix', 'Artist Display Name', 'Artist Display Bio', 'Artist Suffix', 'Artist Alpha Sort', 'Artist Nationality', 'Artist Begin Date', 'Artist End Date', 'Artist Gender', 'Artist ULAN URL', 'Artist Wikidata URL', 'Object Date', 'Credit Line', 'Geography Type', 'City', 'State', 'County', 'Country', 'Region', 'Subregion', 'Locale', 'Locus', 'Excavation', 'River', 'Classification', 'Rights and Reproduction', 'Link Resource', 'Object Wikidata URL', 'Metadata Date', 'Repository', 'Tags_x', 'Tags AAT URL', 'Tags Wikidata URL'], inplace=True)
print(df2.head())
df2.to_csv(Path.cwd() / "Data" / "MetObjWithTags2.csv", index=False)