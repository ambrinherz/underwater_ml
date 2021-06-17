import pickle
import json
import random
import pandas as pd 

TEST_PERCENT = 0.6

with open('ww_no_ll.csv', 'r') as f:
  
    list_of_rows = f.readlines()
    #print(list_of_rows[0])
  
rows = list_of_rows[1:]
random.shuffle(rows)

split = round(TEST_PERCENT * len(rows))

with open('worldwide/worldwide_sample_f60.csv', 'w') as f:
	f.write(list_of_rows[0])
	for x in range(0, split):
		f.write(rows[x]) 

with open('worldwide/worldwide_sample_s40.csv', 'w') as f:
	f.write(list_of_rows[0])
	for x in range(split,len(rows)):
		f.write(rows[x])



