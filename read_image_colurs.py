from pathlib import Path
import pandas as pd
pd.set_option("display.max_columns", None)

metadata = pd.read_csv(Path.cwd() / "Data" / "ImageColoursRaw.csv")
df = pd.DataFrame(metadata)
df["Filename"] = df["Filename"].apply(lambda x: Path(x).name)
df.drop(columns=["Count"], inplace=True)
df = df.pivot_table(index='Filename', columns="Colour", aggfunc='sum', fill_value=0).reset_index()
df.columns = ['Filename' if col[0] == 'Filename' else f'Colour_{col[1]}' for col in df.columns]
df = df.rename(columns={"Filename": "Object ID"})
print(df.head(100))
df.to_csv(Path.cwd() / "Data" / "ImageColoursPivot.csv", index=False)

