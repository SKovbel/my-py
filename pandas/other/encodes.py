import os
import pandas as pd
import numpy as np

# helpful character encoding module
import charset_normalizer

DIR = os.path.dirname(os.path.abspath(__file__))
path = lambda name: os.path.join(DIR, '..', '..', 'data', name)

# set seed for reproducibility
np.random.seed(0)

# start with a string
before = "This is the euro symbol: €"
print(type(before), before)

after = before.encode("utf-8", errors="replace")
before1 = after.decode("utf-8")
print(type(after), after, type(before1), before1)

after2 = before.encode("ascii", errors="replace")
before2 = after2.decode("ascii")
print(type(after2), after2, before2)


# fix enc1
kickstarter_2016 = pd.read_csv(path("encodes.csv"))
with open(path("encodes.csv"), 'rb') as rawdata:
    result = charset_normalizer.detect(rawdata.read(10000))
print(result)

# fix enc2
kickstarter_2016 = pd.read_csv(path("encodes.csv"), encoding='Windows-1252')
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")

# check what the character encoding might be
print(result)
