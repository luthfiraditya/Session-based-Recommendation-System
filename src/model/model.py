import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


# import pandas as pd


class GNN(Module):
    """
    GNN class.

    Args:
        hidden_size (int): The hidden size of the GNN.
        step (int, optional): The number of steps to run the GNN. Defaults to 1.
    """

    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        """
        GNN cell.

        Args:
           A (torch.Tensor): The adjacency matrix.
           hidden (torch.Tensor): The hidden state.

        Returns:
           torch.Tensor: The new hidden state.
        """

        # ================================ Adjacency Transformation ==========================================
        # INPUT GATE rumus (2.1) -> multiplying the adjacency matrix A by the linear transformation of the hidden state hidden.
        # The linear transformation is defined by the parameters self.linear_edge_in and self.b_iah
        # a^t s,i = As,i: [v^t−1 . . . , v^t−1 n]⊤ H+ b,
        # torch.matmul() -> function to matrix multiplication or batched matrix multiplication between two tensors.

        # for incoming
        # A_tensor[:, :, :A_tensor.shape[1]] -> take the incoming part
        # linear_edge_in(hidden) -> applies linear transformation to hidden tensor.
        # b_iah -> bias -> first initiate list (tensor) of zero, with the shape of hidden_size
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        # output shape -> (num_session, num_unique_nodes, hidden_size)

        # for outgoing
        # A[:, :, A.shape[1]: 2 * A.shape[1]] -> take the outgoing part
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # output shape -> (num_session, num_unique_nodes, hidden_size)

        # concatenate the tensors input_in and input_out on third(2) dimension -> (first, second, third)
        inputs = torch.cat([input_in, input_out], 2)  # RUMUS 2.1
        # output shape -> (num_session, num_unique_nodes, hidden_size*2)

        # ================================ Adjacency Transformation ==========================================

        #  computes the linear transformation of the "input"
        gi = F.linear(inputs, self.w_ih, self.b_ih)

        #  computes the linear transformation of the "hidden"
        gh = F.linear(hidden, self.w_hh, self.b_hh)

        i_r, i_u, i_c = gi.chunk(3, 2)  # chunk the result of linear transformation of INPUT
        h_r, h_u, h_c = gh.chunk(3, 2)  # chunk the result of linear transformation of HIDDEN

        updategate = torch.sigmoid(i_u + h_u)  # RUMUS 2.2
        resetgate = torch.sigmoid(i_r + h_r)  # RUMUS 2.3

        candidatestate = torch.tanh(i_c + resetgate * h_c)  # RUMUS 2.4 -> CANDIDATE STATE
        hy = (hidden - candidatestate) * updategate + candidatestate  # RUMUS 2.5
        return hy

    def forward(self, A, hidden):
        """
        Forward pass of the GNN.

        Args:
            A (torch.Tensor): The adjacency matrix.
            hidden (torch.Tensor): The initial hidden state.

        Returns:
            torch.Tensor: The final hidden state.
        """
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    """
    SessionGraph model for session-based recommendation.

    Args:
        opt (argparse.Namespace): Options and hyperparameters.
        n_node (int): Number of nodes/items in the graph.

    Attributes:
        hidden_size (int): The size of the hidden state.
        n_node (int): Number of nodes/items in the graph.
        batch_size (int): The batch size.
        nonhybrid (bool): Whether to use non-hybrid mode.
        embedding (nn.Embedding): Embedding layer.
        gnn (GNN): GNN module.
        linear_one (nn.Linear): Linear layer.
        linear_two (nn.Linear): Linear layer.
        linear_three (nn.Linear): Linear layer.
        linear_transform (nn.Linear): Linear layer.
        linear_t (nn.Linear): Linear layer.
        loss_function (nn.CrossEntropyLoss): Loss function.
        optimizer (torch.optim.Adam): Optimizer.
        scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler.

    Methods:
        reset_parameters(): Reset model parameters.
        compute_scores(hidden, mask): Compute the scores for the recommendations.
        forward(inputs, A): Forward pass of the model.

    """

    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the model parameters.

        Initializes the model parameters with uniform values within the range of (-stdv, stdv),
        where stdv = 1.0 / sqrt(hidden_size).

        Returns:
            None

        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        """
        Compute scores for the recommendations.

        Args:
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, seq_length, hidden_size).
            mask (torch.Tensor): Mask tensor indicating valid items of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Scores tensor of shape (batch_size, n_nodes).

        """

        # ================================== LOCAL EMBEDDING ==================================
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask,
                                                                  1) - 1]  # batch_size x latent_size ----> also use for GLOBAL EMBEDDING
        # ================================== LOCAL EMBEDDING ==================================

        # ================================== GLOBAL EMBEDDING ==================================
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size

        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (b,s,1)
        # alpha = torch.sigmoid(alpha) # B,S,1
        alpha = F.softmax(alpha, 1)  # B,S,1
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # (b,d) # ---> GLOBAL EMBEDDING
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        a = a.view(ht.shape[0], 1, ht.shape[1])
        # ================================== GLOBAL EMBEDDING ==================================

        # ================================== TARGET EMBEDDING ==================================
        # target attention: sigmoid(hidden M b)
        # mask  # batch_size x seq_length
        b = self.embedding.weight[1:]  # n_nodes x latent_size # weight of hidden target

        hidden = hidden * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size
        qt = self.linear_t(hidden)  # hidden of target item # batch_size x seq_length x latent_size

        # beta = torch.sigmoid(b @ qt.transpose(1,2))  # batch_size x n_nodes x seq_length
        beta = F.softmax(b @ qt.transpose(1, 2), -1)  # 2.9  # batch_size x n_nodes x seq_length
        target = beta @ hidden  # 2.10 # batch_size x n_nodes x latent_size # --> TARGET EMBEDDING
        # ================================== TARGET EMBEDDING ==================================

        a = a + target  # b,n,d #concatenation
        scores = torch.sum(a * b, -1)  # b,n
        # scores = F.softmax(scores)
        # scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length).
            A (torch.Tensor): Adjacency matrix tensor of shape (batch_size, n_nodes, n_nodes).

        Returns:
            torch.Tensor: Hidden state tensor of shape (batch_size, seq_length, hidden_size).

        """
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    """
    Move a PyTorch tensor or module to the CUDA device if available.

    Args:
        variable (torch.Tensor or torch.nn.Module): Input tensor or module.

    Returns:
        torch.Tensor or torch.nn.Module: Tensor or module on the CUDA device if available,
            otherwise on the CPU.

    """
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    """
    Move a PyTorch tensor or module to the CPU.

    Args:
        variable (torch.Tensor or torch.nn.Module): Input tensor or module.

    Returns:
        torch.Tensor or torch.nn.Module: Tensor or module on the CPU.

    """
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    """
    Perform a forward pass of the model.

    Args:
        model (SessionGraph): Instance of the SessionGraph model.
        i (int): Index of the data slice.
        data: Data object containing the input data.

    Returns:
        tuple: Tuple containing targets and scores.
            targets (torch.Tensor): Target tensor.
            scores (torch.Tensor): Scores tensor.

    """
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = np.array(A)
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    """
    Train and evaluate the SessionGraph model.

    Args:
        model (SessionGraph): The SessionGraph model.
        train_data (SessionData): Training data.
        test_data (SessionData): Test data.

    Returns:
        float: Hit rate.
        float: Mean Reciprocal Rank (MRR).

    """
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())

        # calculate the loss and optimize the model by update the parameter
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
