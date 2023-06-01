import numpy as np
import csv
import pandas as pd
import random

imageQueryReader = csv.reader(open('pillar_imageNames_query.csv', 'r'))
gtQueryReader = csv.reader(open('pillar_gt_query.csv', 'r'))

rawUtmQ = []
rawQueryIndex = []

i = 0
for line in gtQueryReader:
    rawUtmQ.append(line)
    i += 1
for line in imageQueryReader:
    rawQueryIndex.append(line)

randQueryIndex = np.random.choice(i, i, replace=False)

utmQ = []
queryIndex = []
for line in np.array(rawUtmQ)[randQueryIndex]:
    utmQ.append([float(x) for x in line[:6]])
for line in np.array(rawQueryIndex)[randQueryIndex]:
    queryIndex.append(line)

pd.DataFrame(queryIndex).to_csv('_pillar_imageNames_query.csv', index=False, header=False)


imageIndexReader = csv.reader(open('pillar_imageNames_index.csv', 'r'))
gtIndexReader = csv.reader(open('pillar_gt_database.csv', 'r'))

rawUtmDb = []
rawDbIndex = []

i = 0
for line in gtIndexReader:
    rawUtmDb.append(line)
    i += 1
for line in imageIndexReader:
    rawDbIndex.append(line)

randDbIndex = np.random.choice(i, i, replace=False)

utmDb = []
dbIndex = []
for line in np.array(rawUtmDb)[randDbIndex]:
    utmDb.append([float(x) for x in line[:6]])
for line in np.array(rawDbIndex)[randDbIndex]:
    dbIndex.append(line)

pd.DataFrame(dbIndex).to_csv('_pillar_imageNames_index.csv', index=False, header=False)

np.savez('pillar.npz', utmQ=np.array(utmQ), utmDb=np.array(utmDb), posDistThr=300)
