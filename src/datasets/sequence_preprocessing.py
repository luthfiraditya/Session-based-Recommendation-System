import json
import os
import pickle

# Opening JSON file
f = open('json_folder/tra_sess_SD.json')
g = open('json_folder/tes_sess_SD.json')
h = open('json_folder/sess_clicks_SD.json')
i = open('json_folder/sess_date_SD.json')
# returns JSON object as
# a dictionary
tra_sess = json.load(f)
tes_sess = json.load(g)
sess_clicks = json.load(h)
sess_date = json.load(i)

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print('length of training sequence = ',len(tr_seqs))
print('length of testing sequence = ',len(te_seqs))
print('example of sequence, date and label of training = ',tr_seqs[:10], tr_dates[:10], tr_labs[:10])
print('example of sequence, date and label of training = ',te_seqs[:10], te_dates[:10], te_labs[:10])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

if not os.path.exists('yoochoose1_64_ex'):
    os.makedirs('yoochoose1_64_ex')

pickle.dump(tes, open('yoochoose1_64_ex/test.txt', 'wb'))

split64 = int(len(tr_seqs) / 64)
print(len(tr_seqs[-split64:]))

tra64 = (tr_seqs[-split64:], tr_labs[-split64:])
seq64 = tra_seqs[tr_ids[-split64]:]

pickle.dump(tra64, open('yoochoose1_64_ex/train.txt', 'wb'))
pickle.dump(seq64, open('yoochoose1_64_ex/all_train_seq.txt', 'wb'))
print("length of test : ",len(tes))
print("length of tra64 : ",len(tra64))
print("length of seq64 : ",len(seq64))