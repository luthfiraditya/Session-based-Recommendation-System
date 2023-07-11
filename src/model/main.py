import argparse
import os
import pickle
import time
from utils import build_graph, Data, split_validation, create_csv_file
from model import *
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose_64/sample')
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def main():

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    #all_train_seq = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))

    #output_path = 'datasets/' + opt.dataset + '/cosmograph_'+opt.dataset+'.csv'
    #create_csv_file(all_train_seq, output_path)
    #print(len(all_train_seq))


    # adjacency_path = 'datasets/' + opt.dataset + '/adjacency_' + opt.dataset + '.csv'
    # adjacency_matrix_df = create_adjacency(all_train_seq)
    # adjacency_matrix_df.to_csv(adjacency_path)

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    '''
    batch_size=10
    slices = train_data.generate_batch(batch_size)  # Call the generate_batch() method
    # The generate_batch() method returns slices for batching the data

    for i in slices:
        alias_inputs, A, items, mask, targets = train_data.get_slice(i)  # Call the get_slice() method for each slice
        # Use the returned values for further processing or analysis

    print(A)
    '''


    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310


    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


    # Create the directory if it doesn't exist
    if not os.path.exists('model/' + opt.dataset):
        os.makedirs('model/' + opt.dataset)

    # Save the model
    torch.save(model.state_dict(), 'model/' + opt.dataset + '/checkpoint2_'+opt.dataset+'.pth')


if __name__ == '__main__':
    main()