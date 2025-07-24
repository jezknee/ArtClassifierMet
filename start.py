from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
#pd.set_option("low_memory",False)

path = Path.cwd() / "Data" / "MetObjects.txt"
print(path)
metadata = []
df = pd.read_csv(path, encoding="utf-8")
#print(df.head())
#df.to_csv(Path.cwd() / "Data" / "MetObj.csv", index=False)