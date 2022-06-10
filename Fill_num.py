import csv
import pandas as pd
import re
import numpy as np


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


filename = 'test_01.csv'
rrows = list()
with open(filename, newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        i = len(row)
        for k in range(len(row)):
            if row[k] == ' ?':
                row[k] = ''
            if(is_number(row[k])):
                row[k] = float(row[k])
        rrows.append(row)
    # print(i)
csvfile.close()

l = list(range(i))
for j in range(i):
    l[j] = str(j)
rrows[0][0] = 0.0

with open('writedFile.csv', 'w', newline='') as f:
    writeCsv = csv.writer(f)
    writeCsv.writerow(l)
    writeCsv.writerows(rrows)
f.close()

df = pd.read_csv('writedFile.csv')
for j in range(i-1):
    if df.iat[0, j] == 1:
        df[str(j)] = df[str(j)].fillna(df[str(j)].mean())
    else:
        df[str(j)] = df[str(j)].fillna(df[str(j)].mode().iloc[0])

df.to_csv('new' + filename)
