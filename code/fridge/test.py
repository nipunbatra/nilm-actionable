import sys
import os

print os.getcwd()
print sys.argv[1], sys.argv[2]
a, b = sys.argv[1], sys.argv[2]
with open(a+b+".txt", "w") as f:
    f.write(a+b)