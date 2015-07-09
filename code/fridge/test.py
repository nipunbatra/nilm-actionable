import sys
import os

print os.getcwd()
print sys.argv[1], sys.argv[2]
a, b = sys.argv[1], sys.argv[2]

import pandas as pd

df = pd.HDFStore(a)
print df.head()