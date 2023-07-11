import json
import operator

# Opening JSON file
f = open('json_folder/sess_clicks_RG.json')
g = open('json_folder/sess_date_RG.json')
# returns JSON object as
# a dictionary
sess_clicks = json.load(f)
sess_date = json.load(g)

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

def count_unique_ids(dictionary):
  unique_ids = set()
  for values in dictionary.values():
    for id in values:
      unique_ids.add(id)
  return len(unique_ids)

number_of_unique_ids = count_unique_ids(sess_clicks)
number_of_session=len(sess_clicks)


print(f'number of unique item id after data filtering = {number_of_unique_ids}')
print(f'number of session = {number_of_session}')


# Open a file for writing
with open("json_folder/sess_clicks_DF.json", "w") as f:
    # Write the dict to the file
    json.dump(sess_clicks, f, indent=4)

# Open a file for writing
with open("json_folder/sess_date_DF.json", "w") as f:
    # Write the dict to the file
    json.dump(sess_date, f, indent=4)