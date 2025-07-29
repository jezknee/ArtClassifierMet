import pandas as pd
from pathlib import Path

meta_path = Path.cwd() / "Data" / "MetObjects.txt"
metadata = pd.read_csv(meta_path)
df = pd.DataFrame(metadata["object_id", "department", "tags"])
print(df.head())

