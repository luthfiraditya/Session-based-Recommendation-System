import csv
#import pandas as pd
import numpy as np
import networkx as nx

def build_graph(train_data):
    """Builds a directed graph based on the training data.

    Args:
        train_data (list): List of sequences representing training data.

    Returns:
        networkx.DiGraph: Directed graph representing the training data.
    """
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


'''
def create_adjacency(train_data):
    """Creates an adjacency matrix based on the training data.

    Args:
        train_data (list): List of sequences representing training data.

    Returns:
        pandas.DataFrame: Adjacency matrix representing the training data.
    """

    graph = nx.DiGraph()
    nodes = set()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.has_edge(seq[i], seq[i + 1]):
                weight = graph[seq[i]][seq[i + 1]]['weight'] + 1
            else:
                weight = 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
            nodes.add(seq[i])
            nodes.add(seq[i + 1])

    node_indices = {node: index for index, node in enumerate(nodes)}
    num_nodes = len(nodes)
    adjacency_matrix_out = np.zeros((num_nodes, num_nodes))
    adjacency_matrix_in = np.zeros((num_nodes, num_nodes))

    for u, v, weight in graph.edges(data='weight'):
        u_idx = node_indices[u]
        v_idx = node_indices[v]
        adjacency_matrix_out[u_idx][v_idx] = weight

    for v, u, weight in graph.in_edges(data='weight'):
        u_idx = node_indices[u]
        v_idx = node_indices[v]
        adjacency_matrix_in[u_idx][v_idx] = weight

    adjacency_matrix_out_sum = np.sum(adjacency_matrix_out, axis=1, keepdims=True)
    adjacency_matrix_in_sum = np.sum(adjacency_matrix_in, axis=1, keepdims=True)

    adjacency_matrix_out_sum[adjacency_matrix_out_sum == 0] = 1
    adjacency_matrix_in_sum[adjacency_matrix_in_sum == 0] = 1

    adjacency_matrix_out /= adjacency_matrix_out_sum
    adjacency_matrix_in /= adjacency_matrix_in_sum

    adjacency_matrix = np.concatenate((adjacency_matrix_out, adjacency_matrix_in), axis=1)

    node_ids = sorted(nodes)
    df_adjacency = pd.DataFrame(adjacency_matrix, index=node_ids, columns=node_ids * 2)
    df_adjacency.fillna(0, inplace=True)

    return df_adjacency
'''

def create_csv_file(all_train_seq, output_path):
    """Creates a CSV file from the training sequences.

    Args:
        all_train_seq (list): List of training sequences.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', newline='') as g:
        writer = csv.writer(g)
        writer.writerow(['source', 'target'])
        for seq in all_train_seq:
            for i in range(len(seq) - 1):
                writer.writerow([seq[i], seq[i + 1]])


def data_masks(all_usr_pois, item_tail):
    """Creates masks for the input data.

    Args:
        all_usr_pois (list): List of user sequences.
        item_tail (list): Tail items used for padding.

    Returns:
        tuple: A tuple containing the padded user sequences, masks, and maximum length.
    """
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    """Splits the training set into training and validation sets.

    Args:
        train_set (tuple): A tuple containing training inputs and targets.
        valid_portion (float): Proportion of the data to use for validation.

    Returns:
        tuple: A tuple containing the training and validation sets.
    """
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        """Generates batches of data.

           Args:
               batch_size (int): Size of each batch.

           Returns:
               list: List of slices containing indices for each batch.
           """
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        """
            Retrieve a slice of data at index i.

            Args:
                i (int): Index of the slice to retrieve.

            Returns:
                tuple: A tuple containing alias_inputs, A, items, mask, and targets.

            """
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            # ceate adjacency matrix
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                # u dan v untuk dapetin baris dan kolomnya
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
