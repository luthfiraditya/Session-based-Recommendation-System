import json
import operator
import datetime

# Opening JSON file
f = open('json_folder/sess_clicks_DF.json')
g = open('json_folder/sess_date_DF.json')
# returns JSON object as
# a dictionary
sess_clicks = json.load(f)
sess_date = json.load(g)


# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print('lenght of training session = ',len(tra_sess))    # 186670    # 7966257
print('length of testing session = ',len(tes_sess))    # 15979     # 15324
print('example of training session = ',tra_sess[:3])
print('example of training session = ',tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Open a file for writing
with open("json_folder/sess_clicks_SD.json", "w") as f:
    # Write the dict to the file
    json.dump(sess_clicks, f, indent=4)

# Open a file for writing
with open("json_folder/sess_date_SD.json", "w") as f:
    # Write the dict to the file
    json.dump(sess_date, f, indent=4)

# Open a file for writing
with open("json_folder/tra_sess_SD.json", "w") as f:
    # Write the dict to the file
    json.dump(tra_sess, f, indent=4)

# Open a file for writing
with open("json_folder/tes_sess_SD.json", "w") as f:
    # Write the dict to the file
    json.dump(tes_sess, f, indent=4)