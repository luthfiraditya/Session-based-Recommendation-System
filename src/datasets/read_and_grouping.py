import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import json

dataset = 'yoochoose_data/yoochoose-clicks.dat'
f = open(dataset, "r")
reader = csv.DictReader(f, delimiter=',')

sess_clicks = {}
sess_date = {}
ctr = 0
curid = -1
curdate = None

for data in reader:
    sessid = data['session_id']
    if curdate and not curid == sessid:
        date = ''
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S')) #convert date in the format YYYY-MM-DDTHH:MM:SS into Unix timestamp
        sess_date[curid] = date
    curid = sessid
    item = data['item_id']
    curdate = ''
    curdate = data['timestamp']

    if sessid in sess_clicks:
        sess_clicks[sessid] += [item]
    else:
        sess_clicks[sessid] = [item]
    ctr += 1

date = ''
date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
sess_date[curid] = date


def count_unique_ids(dictionary):
  unique_ids = set()
  for values in dictionary.values():
    for id in values:
      unique_ids.add(id)
  return len(unique_ids)

number_of_unique_ids = count_unique_ids(sess_clicks)
number_of_session=len(sess_clicks)
number_of_date = len(sess_date)

print(f'number of unique item id at the beginning = {number_of_unique_ids}')
print(f'number of session = {number_of_session}')
print(f'number of date = {number_of_date}')


# Open a file for writing
with open("json_folder/sess_clicks_RG.json", "w") as f:
    # Write the dict to the file
    json.dump(sess_clicks, f, indent=4)

# Open a file for writing
with open("json_folder/sess_date_RG.json", "w") as f:
    # Write the dict to the file
    json.dump(sess_date, f, indent=4)



