import argparse
import os

import dgl
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from scipy import sparse
import scipy.sparse as sp


def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    norm = torch.pow(in_deg, -0.5)
    norm[torch.isinf(norm)] = 0
    # in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    # norm = 1.0 / in_deg
    return norm


def get_big_graph(backgrounds, num_ents, num_rels):
    if (len(backgrounds) == 0):
        src, dst = np.arange(num_ents), np.arange(num_ents)
        rel = np.zeros(num_ents)
    else:
        # if len(backgrounds) == 0:
        #     return dgl.DGLGraph()
        data = backgrounds
        src, rel, dst = data.transpose()

        loop_nodes = np.arange(num_ents)
        src, dst = np.concatenate((src, dst, loop_nodes)), np.concatenate((dst, src, loop_nodes))
        vec_zeros = np.zeros(num_ents)
        rel = np.concatenate((rel, rel + num_rels, vec_zeros))

    g = dgl.DGLGraph()
    g.add_nodes(num_ents)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_ents, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.edata['rel'] = torch.LongTensor(rel)
    # g.edata['type_o'] = torch.LongTensor(rel_o)
    return g

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).cuda()

def make_batch4(data, s_frequency, o_frequency, s_label, o_label, times, batch_size, err_mat_list, err_mat_inv_list):
    for i in range(len(times)):
        l = times[i][0]
        if i < len(times) - 1:
            r = times[i + 1][0]
        else:
            r = len(data)
        yield [data[l:r], s_frequency[l:r], o_frequency[l:r], s_label[l:r], o_label[l:r], sparse_mx_to_torch_sparse_tensor(err_mat_list[i]), sparse_mx_to_torch_sparse_tensor(err_mat_inv_list[i])]


def make_batch(data, s_frequency, o_frequency, s_label, o_label, times, batch_size):
    for i in range(len(times)):
        l = times[i][0]
        if i < len(times) - 1:
            r = times[i + 1][0]
        else:
            r = len(data)
        yield [data[l:r], s_frequency[l:r], o_frequency[l:r], s_label[l:r], o_label[l:r]]


def make_batch2(data, s_history, o_history, s_frequency, o_frequency, s_label, o_label, times, batch_size):
    for i in range(len(times)):
        l = times[i][0]
        if i < len(times) - 1:
            r = times[i + 1][0]
        else:
            r = len(data)
        yield [data[l:r], s_history[l:r], o_history[l:r], s_frequency[l:r], o_frequency[l:r], s_label[l:r],
               o_label[l:r]]

def make_batch3(data, s_frequency, o_frequency, s_label, o_label, times, batch_size, super_glist, glist):
    for i in range(len(times)):
        l = times[i][0]
        if i < len(times)-1:
            r = times[i+1][0]
        else:
            r = len(data)
        if i < 2:
            yield [data[l:r], s_frequency[l:r], o_frequency[l:r], s_label[l:r], o_label[l:r], super_glist[:i], glist[:i]]
        else:
            yield [data[l:r], s_frequency[l:r], o_frequency[l:r], s_label[l:r], o_label[l:r], super_glist[i-2:i], glist[i-2:i]]

def execute_valid_oracle(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                         data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, s_history, o_history):
    s_ranks1 = []
    o_ranks1 = []
    all_ranks1 = []
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    # total_data = utils.to_device(torch.from_numpy(total_data))
    for batch_data in make_batch2(data, s_history, o_history, s_frequency, o_frequency, valid_s_label, valid_o_label,
                                  dev_t, args.batch_size):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()
        batch_data[5] = torch.from_numpy(batch_data[5]).to(device).float()
        batch_data[6] = torch.from_numpy(batch_data[6]).to(device).float()

        with torch.no_grad():
            sub_rank1, obj_rank1, _, ce_all_acc = model(batch_data, his_g, 'Valid_Oracle', total_data)

            s_ranks1 += sub_rank1
            o_ranks1 += obj_rank1
            tmp1 = sub_rank1 + obj_rank1
            all_ranks1 += tmp1

    return s_ranks1, o_ranks1, all_ranks1


def execute_valid(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                  data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    valid_loss = 0
    batch_num = 0
    # s_ranks, o_ranks, all_ranks = [], [], []
    for batch_data in make_batch(data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, args.batch_size):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()

        with torch.no_grad():
            cur_loss = model(batch_data, his_g, 'Valid', total_data)
            # sub_rank, obj_rank, cur_loss = model(batch_data, his_g, 'Valid', total_data)
            valid_loss += cur_loss.item()
            batch_num += 1
            # s_ranks += sub_rank
            # o_ranks += obj_rank
            # tmp = sub_rank + obj_rank
            # all_ranks += tmp

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return valid_loss / batch_num
    # return s_ranks, o_ranks, all_ranks, valid_loss / batch_num

def execute_valid4(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                  data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, err_mat, err_inv_mat):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    valid_loss = 0
    batch_num = 0
    # s_ranks, o_ranks, all_ranks = [], [], []
    for batch_data in make_batch4(data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, args.batch_size, err_mat, err_inv_mat):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()

        with torch.no_grad():
            cur_loss = model(batch_data, his_g, 'Valid', total_data)
            # sub_rank, obj_rank, cur_loss = model(batch_data, his_g, 'Valid', total_data)
            valid_loss += cur_loss.item()
            batch_num += 1
            # s_ranks += sub_rank
            # o_ranks += obj_rank
            # tmp = sub_rank + obj_rank
            # all_ranks += tmp

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return valid_loss / batch_num

def execute_valid5(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                  data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, err_mat, err_inv_mat, his_data, train_t):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    valid_loss = 0
    batch_num = 0
    last_data = make_batch_only_data(his_data, train_t, args.batch_size)
    list_diff_data = [torch.tensor(last_data[-args.max_len]).to(device)]
    # s_ranks, o_ranks, all_ranks = [], [], []
    for batch_data in make_batch4(data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, args.batch_size, err_mat, err_inv_mat):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()

        with torch.no_grad():
            cur_loss = model(batch_data, his_g, list_diff_data, 'Valid', total_data)
            # sub_rank, obj_rank, cur_loss = model(batch_data, his_g, 'Valid', total_data)
            valid_loss += cur_loss.item()
            batch_num += 1
            # s_ranks += sub_rank
            # o_ranks += obj_rank
            # tmp = sub_rank + obj_rank
            # all_ranks += tmp

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        list_diff_data.append(batch_data[0])
        list_diff_data = list_diff_data[-args.max_len:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return valid_loss / batch_num


def execute_valid2(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                   data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, s_history, o_history):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()
    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []
    valid_loss = 0
    batch_num = 0
    # s_ranks, o_ranks, all_ranks = [], [], []
    for batch_data in make_batch2(data, s_history, o_history, s_frequency, o_frequency, valid_s_label, valid_o_label,
                                  dev_t, args.batch_size):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()
        batch_data[5] = torch.from_numpy(batch_data[5]).to(device).float()
        batch_data[6] = torch.from_numpy(batch_data[6]).to(device).float()

        with torch.no_grad():
            sub_rank1, obj_rank1, cur_loss1, \
                sub_rank2, obj_rank2, cur_loss2, \
                sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, his_g, 'Valid', total_data)
            # sub_rank, obj_rank, cur_loss = model(batch_data, his_g, 'Valid', total_data)
            # sub_rank1, obj_rank1, cur_loss1, \
            #     sub_rank2, obj_rank2, cur_loss2, \
            #      ce_all_acc = model(batch_data, his_g, 'Valid', total_data)
            cur_loss = (cur_loss1 + cur_loss2) / 2
            valid_loss += cur_loss.item()
            batch_num += 1
            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return valid_loss / batch_num, s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3
    # return s_ranks, o_ranks, all_ranks, valid_loss / batch_num

def execute_valid3(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                  data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, super_glist, glist):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    valid_loss = 0
    batch_num = 0
    # s_ranks, o_ranks, all_ranks = [], [], []
    for batch_data in make_batch3(data, s_frequency, o_frequency, valid_s_label, valid_o_label, dev_t, args.batch_size, super_glist, glist):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()

        with torch.no_grad():
            cur_loss = model(batch_data, his_g, 'Valid', total_data)
            # sub_rank, obj_rank, cur_loss = model(batch_data, his_g, 'Valid', total_data)
            valid_loss += cur_loss.item()
            batch_num += 1
            # s_ranks += sub_rank
            # o_ranks += obj_rank
            # tmp = sub_rank + obj_rank
            # all_ranks += tmp

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return valid_loss / batch_num
def execute_test(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                 data, s_frequency, o_frequency, test_s_label, test_o_label, test_t):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    s_ranks, o_ranks, all_ranks = [], [], []

    for batch_data in make_batch(data, s_frequency, o_frequency, test_s_label, test_o_label, test_t, args.batch_size):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()

        with torch.no_grad():
            sub_rank, obj_rank = model(batch_data, his_g, 'Test', total_data)

            s_ranks += sub_rank
            o_ranks += obj_rank
            tmp = sub_rank + obj_rank
            all_ranks += tmp

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return s_ranks, o_ranks, all_ranks


def execute_test2(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                  data, s_frequency, o_frequency, test_s_label, test_o_label, test_t, s_history, o_history):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()
    s_ranks1 = []
    o_ranks1 = []
    all_ranks1 = []

    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []
    s_ranks, o_ranks, all_ranks = [], [], []

    for batch_data in tqdm(
            make_batch2(data, s_history, o_history, s_frequency, o_frequency, test_s_label, test_o_label, test_t,
                        args.batch_size), total=len(test_t), desc='Test', dynamic_ncols=True):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()
        batch_data[5] = torch.from_numpy(batch_data[5]).to(device).float()
        batch_data[6] = torch.from_numpy(batch_data[6]).to(device).float()

        with torch.no_grad():
            sub_rank1, obj_rank1, cur_loss1, \
                sub_rank2, obj_rank2, cur_loss2, \
                sub_rank3, obj_rank3, cur_loss3, \
                ce_all_acc = model(batch_data, his_g, 'Test', total_data)

            s_ranks1 += sub_rank1
            o_ranks1 += obj_rank1
            tmp1 = sub_rank1 + obj_rank1
            all_ranks1 += tmp1

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return s_ranks1, o_ranks1, all_ranks1, \
        s_ranks2, o_ranks2, all_ranks2, \
        s_ranks3, o_ranks3, all_ranks3,

def execute_test3(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                 data, s_frequency, o_frequency, test_s_label, test_o_label, test_t, super_glist, glist):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    s_ranks, o_ranks, all_ranks = [], [], []

    for batch_data in make_batch3(data, s_frequency, o_frequency, test_s_label, test_o_label, test_t, args.batch_size, super_glist, glist):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()

        with torch.no_grad():
            sub_rank, obj_rank = model(batch_data, his_g, 'Test', total_data)

            s_ranks += sub_rank
            o_ranks += obj_rank
            tmp = sub_rank + obj_rank
            all_ranks += tmp

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return s_ranks, o_ranks, all_ranks

def execute_test4(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                 data, s_frequency, o_frequency, test_s_label, test_o_label, test_t, err_mat, err_inv_mat):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    s_ranks, o_ranks, all_ranks = [], [], []

    for batch_data in make_batch4(data, s_frequency, o_frequency, test_s_label, test_o_label, test_t, args.batch_size, err_mat, err_inv_mat):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()

        with torch.no_grad():
            sub_rank, obj_rank = model(batch_data, his_g, 'Test', total_data)

            s_ranks += sub_rank
            o_ranks += obj_rank
            tmp = sub_rank + obj_rank
            all_ranks += tmp

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return s_ranks, o_ranks, all_ranks

def make_batch_only_data(data,times, batch_size):
    output = []
    for i in range(len(times)):
        l = times[i][0]
        if i < len(times) - 1:
            r = times[i + 1][0]
        else:
            r = len(data)
        output.append(data[l:r])
    return output

def execute_test5(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                 data, s_frequency, o_frequency, test_s_label, test_o_label, test_t, err_mat, err_inv_mat, his_data, train_t):
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()

    s_ranks, o_ranks, all_ranks = [], [], []
    last_data = make_batch_only_data(his_data, train_t, args.batch_size)
    list_diff_data = [torch.tensor(last_data[-args.max_len]).to(device)]
    for batch_data in make_batch4(data, s_frequency, o_frequency, test_s_label, test_o_label, test_t, args.batch_size, err_mat, err_inv_mat):
        triples = batch_data[0][:, :3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).to(device).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).to(device).float()

        with torch.no_grad():
            sub_rank, obj_rank = model(batch_data, his_g, list_diff_data, 'Test', total_data)

            s_ranks += sub_rank
            o_ranks += obj_rank
            tmp = sub_rank + obj_rank
            all_ranks += tmp

        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        list_diff_data.append(batch_data[0])
        list_diff_data = list_diff_data[-args.max_len:]
        g = g.to(device)
        his_g.append(g)

    return s_ranks, o_ranks, all_ranks

def write2file(s_ranks, o_ranks, all_ranks, file_test):
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    print("Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    print("Subject test MR (lk): {:.6f}".format(s_mr_lk))
    file_test.write("Subject test MRR (lk): {:.6f}".format(s_mrr_lk) + '\n')
    file_test.write("Subject test MR (lk): {:.6f}".format(s_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        print("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))
        file_test.write("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk) + '\n')

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    print("Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    print("Object test MR (lk): {:.6f}".format(o_mr_lk))
    file_test.write("Object test MRR (lk): {:.6f}".format(o_mrr_lk) + '\n')
    file_test.write("Object test MR (lk): {:.6f}".format(o_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        print("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))
        file_test.write("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk) + '\n')

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    print("ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    print("ALL test MR (lk): {:.6f}".format(all_mr_lk))
    file_test.write("ALL test MRR (lk): {:.6f}".format(all_mrr_lk) + '\n')
    file_test.write("ALL test MR (lk): {:.6f}".format(all_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        print("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))
        file_test.write("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk) + '\n')


def write2file2(s_ranks, o_ranks, all_ranks, file_test):
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    print("Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    print("Subject test MR (lk): {:.6f}".format(s_mr_lk))
    file_test.write("Subject test MRR (lk): {:.6f}".format(s_mrr_lk) + '\n')
    file_test.write("Subject test MR (lk): {:.6f}".format(s_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        print("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))
        file_test.write("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk) + '\n')

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    print("Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    print("Object test MR (lk): {:.6f}".format(o_mr_lk))
    file_test.write("Object test MRR (lk): {:.6f}".format(o_mrr_lk) + '\n')
    file_test.write("Object test MR (lk): {:.6f}".format(o_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        print("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))
        file_test.write("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk) + '\n')

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    print("ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    print("ALL test MR (lk): {:.6f}".format(all_mr_lk))
    file_test.write("ALL test MRR (lk): {:.6f}".format(all_mrr_lk) + '\n')
    file_test.write("ALL test MR (lk): {:.6f}".format(all_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        print("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))
        file_test.write("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk) + '\n')
    return all_mrr_lk


def print2file(s_ranks, o_ranks, all_ranks, file_test):
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    print("Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    print("Subject test MR (lk): {:.6f}".format(s_mr_lk))
    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        print("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    print("Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    print("Object test MR (lk): {:.6f}".format(o_mr_lk))

    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        print("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    print("ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    print("ALL test MR (lk): {:.6f}".format(all_mr_lk))

    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        print("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))


def build_candidate_subgraph(
        num_nodes: int,
        total_triples: np.array,
        total_obj_logit: torch.Tensor,
        k: int,
        num_partitions: int,
) -> dgl.DGLGraph:
    # if pred_sub:
    #     total_obj = total_triples[0]
    #     # total_sub = total_sub_emb[total_triples[:, 0]].unsqueeze(1)
    #     total_rel = total_triples[1]
    #     # total_rel = total_rel_emb[total_triples[:, 1]].unsqueeze(1)

    #     num_queries = total_obj.size(0)
    #     # k = int(num_queries/2)
    #     _, total_topk_sub = torch.topk(total_obj_logit, k=k)
    #     rng = torch.Generator().manual_seed(1234)
    #     total_indices = torch.randperm(num_queries, generator=rng)

    #     graph_list = []
    #     for indices in torch.tensor_split(total_indices, num_partitions):
    #         topk_sub = total_topk_sub[indices]
    #         obj = torch.repeat_interleave(total_obj[indices], k)
    #         rel = torch.repeat_interleave(total_rel[indices], k)
    #         sub = topk_sub.view(-1)
    #         graph = dgl.graph(
    #             (sub, obj),
    #             num_nodes=num_nodes,
    #             device=total_obj.device,
    #         )
    #         graph.ndata["eid"] = torch.arange(num_nodes, device=graph.device)
    #         graph.edata["rid"] = rel
    #         norm = comp_deg_norm(graph)
    #         graph.ndata['norm'] = norm.view(-1, 1)
    #         # graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    #         graph_list.append(graph)
    # else:
    total_sub = total_triples[0]
    # total_sub = total_sub_emb[total_triples[:, 0]].unsqueeze(1)
    total_rel = total_triples[1]
    # total_rel = total_rel_emb[total_triples[:, 1]].unsqueeze(1)

    num_queries = total_sub.size(0)
    # k = int(num_queries/2)
    _, total_topk_obj = torch.topk(total_obj_logit, k=k)
    rng = torch.Generator().manual_seed(1234)
    total_indices = torch.randperm(num_queries, generator=rng)

    graph_list = []
    for indices in torch.tensor_split(total_indices, num_partitions):
        topk_obj = total_topk_obj[indices]
        sub = torch.repeat_interleave(total_sub[indices], k)
        rel = torch.repeat_interleave(total_rel[indices], k)
        obj = topk_obj.view(-1)
        graph = dgl.graph(
            (sub, obj),
            num_nodes=num_nodes,
            device=total_sub.device,
        )
        graph.ndata["eid"] = torch.arange(num_nodes, device=graph.device)
        graph.edata["rid"] = rel
        norm = comp_deg_norm(graph)
        graph.ndata['norm'] = norm.view(-1, 1)
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        graph_list.append(graph)
    return dgl.batch(graph_list)

def build_super_g(num_rels, rel_head, rel_tail, use_cuda, segnn, gpu):
    '''
    :param num_rels: 所有边（不包含反向边）
    :param rel_head: 一个时间戳子图中相同h和r的事实数目统计, (num_rels*2, num_ent) 与特定关系（边）相连的头实体（数目）统计
    :param rel_tail: 一个时间戳子图中相同r和t的事实数目统计, (num_rels*2, num_ent) 与特定关系（边）相连的尾实体（数目）统计
    :param use_cuda: 是否使用GPU
    :param segnn: 是否使用segnn
    :param gpu: GPU的设备号
    :return:
    '''
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float() # 图中每一个节点的入度
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1 # 入度为0的赋值为1
        norm = 1.0 / in_deg # 归一化操作 1/入度
        return norm

    # 关系邻接矩阵（对应不同的方向（位置）模式）
    tail_head = torch.matmul(rel_tail, rel_head.T) # (num_rels*2, num_rels*2) 如果rel1的尾实体是rel2的头实体，那么矩阵相乘的结果(rel1, rel2)对应的值非0
    head_tail = torch.matmul(rel_head, rel_tail.T) # (num_rels*2, num_rels*2) 如果rel1的头实体是rel2的尾实体，那么矩阵相乘的结果(rel1, rel2)对应的值非0
    # torch.diag: 变换生成对角矩阵 torch.diag(torch.sum(rel_tail, axis=1): 与rel相连的尾实体个数
    # 如果rel1的尾实体是rel2的尾实体，那么矩阵相乘的结果(rel1, rel2)对应的值非0
    # 减法运算使对角线元素为0（除去关系自身(rel, rel)的情况），但是如果相同的关系对应多个尾实体（多对一关系），那么仍然会有记录
    tail_tail = torch.matmul(rel_tail, rel_tail.T) - torch.diag(torch.sum(rel_tail, axis=1)) # (num_rels*2, num_rels*2)
    # 如果rel1的头实体是rel2的头实体，那么矩阵相乘的结果(rel1, rel2)对应的值非0
    # 减法运算使对角线元素为0（除去关系自身(rel, rel)的情况），但是如果相同的关系对应多个头实体（一对多关系），那么仍然会有记录
    # 以上操作也就是对一对多关系和多对一关系加上自环
    head_head = torch.matmul(rel_head, rel_head.T) - torch.diag(torch.sum(rel_head, axis=1)) # (num_rels*2, num_rels*2)

    # construct super relation graph from adjacency matrix
    src = torch.LongTensor([])
    dst = torch.LongTensor([])
    p_rel = torch.LongTensor([])
    for p_rel_idx, mat in enumerate([tail_head, head_tail, tail_tail, head_head]): # p_rel_idx: 0, 1, 2, 3
        sp_mat = sparse.coo_matrix(mat) # mat: 每一种类型的关系矩阵
        src = torch.cat([src, torch.from_numpy(sp_mat.row)]) # 行索引数组 对应num_rels邻接矩阵的行关系坐标
        dst = torch.cat([dst, torch.from_numpy(sp_mat.col)]) # 列索引数组 对应num_rels邻接矩阵的列关系坐标
        p_rel = torch.cat([p_rel, torch.LongTensor([p_rel_idx] * len(sp_mat.data))]) # 4类超关系的数目，4类超关系重复多少次

    # 生成super_relation_g的时序子图下的所有事实三元组
    src_tris = src.unsqueeze(1)
    dst_tris = src.unsqueeze(1)
    p_rel_tris = p_rel.unsqueeze(1)
    super_triples = torch.cat((src_tris, p_rel_tris, dst_tris), dim=1).numpy()
    src = src.numpy()
    dst = dst.numpy()
    p_rel = p_rel.numpy()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    p_rel = np.concatenate((p_rel, p_rel + 4))

    # 构造super_relation_g的DGL对象
    if segnn:
        super_g = dgl.graph((src, dst), num_nodes=num_rels*2)
        super_g.edata['rel_id'] = torch.LongTensor(p_rel)
    else:
        super_g = dgl.DGLGraph()
        super_g.add_nodes(num_rels*2) # 加入所有边节点
        super_g.add_edges(src, dst) # 加入所有位置超关系
        norm = comp_deg_norm(super_g) # 对一个子图中的所有节点进行归一化
        rel_node_id = torch.arange(0, num_rels*2, dtype=torch.long).view(-1, 1) # [0, num_rels*2)
        super_g.ndata.update({'id': rel_node_id, 'norm': norm.view(-1, 1)}) # shape都为(num_rels*2, 1)
        super_g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']}) # 更新边, 边的归一化系数为头尾节点的归一化系数相乘
        super_g.edata['type'] = torch.LongTensor(p_rel) # 边的类型数据

    uniq_super_r, r_len, r_to_e = r2e_super(super_triples, num_rels)  # uniq_r: 在当前时间戳内出现的所有的边(包括反向边)；r_len: 记录和边r相关的node的idx范围; e_idx: 和边r相关的node列表
    super_g.uniq_super_r = uniq_super_r  # 在当前时间戳内出现的所有的边(包括反向边)
    super_g.r_to_e = r_to_e  # 和边r相关的node列表，按照uniq_r中记录边的顺序排列
    super_g.r_len = r_len

    # if use_cuda:
    #     super_g.to(gpu)
    #     super_g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return super_g # 通过关系邻接矩阵构建关系超图DGL对象: [[rel1, meta-rel, rel2], ...]

def r2e_super(triplets, num_rels): # triplets(array): [[s, r, o], [s, r, o], ...]
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_super_r = np.unique(rel) # 从小到大排列
    # uniq_r = np.concatenate((uniq_r, uniq_r+num_rels)) # 在当前时间戳内出现的所有的边
    # generate r2e
    r_to_e = defaultdict(set) # 获得和每一条边相关的节点
    for j, (src, rel, dst) in enumerate(triplets): # 对于时间戳内的每一个事实三元组
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        # r_to_e[rel+num_rels].add(src)
        # r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_super_r: # 对于在该时间戳内出现的每一条超边
        r_len.append((idx,idx+len(r_to_e[r]))) # 记录和边r相关的node的idx范围
        e_idx.extend(list(r_to_e[r])) # 和边r相关的node列表
        idx += len(r_to_e[r])
    return uniq_super_r, r_len, e_idx

def mkdir_if_not_exist(file_name):
    import os

    dir_name = os.path.dirname(file_name)  # 返回文件所在的目录路径
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def split_by_time2(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)): # np.array([[s, r, o, time], []...])
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t: # 同一时刻发生的三元组，如果时间戳发生变化，代表进入下一个时间戳
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy()) # 将上一个时间戳的所有事实加入snapshot_list
                snapshots_num += 1
            snapshot = []
        # snapshot.append(train[:3])  # 这样切片得到的是三元组 (s, r, o)
        snapshot.append(train[:])  # 这样切片得到的是四元组 (s, r, o, t)
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy()) # 加入最后一个snapshot
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list: # 对于每一个时间戳中的所有事实 np.array([[s, r, o], [], ...])
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True) # uniq_v: 一个时间戳内从小到大排序的非重复实体列表np.array；edges: relabel的[头实体, 尾实体]，采用uniq_v的index
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1)) # 重新组织为头实体->尾实体的形式
        nodes.append(len(uniq_v)) # 每一个时间戳内出现过的实体数目
        rels.append(len(uniq_r)*2) # 所有无向edges，一正一反，每一个时间戳内出现过的关系数目

    times = set()
    for triple in data:
        times.add(triple[3])
    times = list(times)
    times.sort()

    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    # return snapshot_list # 按时间戳划分的array：[[[s, r, o], [], ...], [], ...]
    return snapshot_list, np.asarray(times)

def build_sub_graph2(num_nodes, num_rels, triples, use_cuda, segnn, gpu):
    '''
    :param num_nodes:
    :param num_rels:
    :param triples: 一个历史时间戳的所有事实三元组 [[s, r, o], [s, r, o], ...] or [[s, r, o, t], [s, r, o, t], ...]
    :param use_cuda:
    :param segnn: 是否使用segnn
    :param gpu:
    :return:
    '''
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float() # 图中每一个节点的入度
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1 # 入度为0的赋值为1
        norm = 1.0 / in_deg # 归一化操作 1/入度
        return norm

    if triples.shape[1] > 3:
        triples = triples[:, :3]
    src, rel, dst = triples.transpose() # (3 * 事实个数)
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src)) # knowledge graph 无向图, 加入反关系
    rel = np.concatenate((rel, rel + num_rels)) # 关系+反关系

    if segnn:
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata['rel_id'] = torch.LongTensor(rel)
    else:
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes) # 加入所有节点
        g.add_edges(src, dst) # 加入所有边
        norm = comp_deg_norm(g) # 对一个子图中的所有节点进行归一化
        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1) # [0, num_nodes)
        g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)}) # shape都为(num_nodes, 1)
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']}) # 更新边, 边的归一化系数为头尾节点的归一化系数相乘
        g.edata['type'] = torch.LongTensor(rel) # 边的类型数据

    uniq_r, r_len, r_to_e = r2e(triples, num_rels) # uniq_r: 在当前时间戳内出现的所有的边(包括反向边)；r_len: 记录和边r相关的node的idx范围; e_idx: 和边r相关的node列表
    g.uniq_r = uniq_r # 在当前时间戳内出现的所有的边(包括反向边)
    g.r_to_e = r_to_e # 和边r相关的node列表，按照uniq_r中记录边的顺序排列
    g.r_len = r_len # 记录和边r相关的node在r_to_e列表中的idx范围，也与uniq_r中边的顺序保持一致
    # if use_cuda:
    #     g.to(gpu)
    #     g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g

def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel+num_rels].add(src)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx
def get_relhead_reltal(tris, num_nodes, num_rels):
    '''
    考虑加入反关系形成无向图
    :param tris: 一个时间戳子图内的所有事实三元组array(/列表) [[s, p, o], ...] or [[s, r, o, t], ...]
    :param num_nodes: 所有的实体数目
    :param num_rels: 所有的关系数目
    :return:
    '''
    if tris.shape[1] > 3:
        tris = tris[:, :3]  # 切片取前三列 s, r, o
    inverse_triplets = tris[:, [2, 1, 0]]
    inverse_triplets[:, 1] = inverse_triplets[:, 1] + num_rels # 将逆关系换成逆关系的id
    all_tris = np.concatenate((tris, inverse_triplets), axis=0)

    rel_head = torch.zeros((num_rels*2, num_nodes), dtype=torch.int) # 二维tensor
    rel_tail = torch.zeros((num_rels*2, num_nodes), dtype=torch.int)

    for tri in all_tris: # 对于support set中的每一个事实三元组
        h, r, t = tri

        rel_head[r, h] += 1 # support中相同h和r的事实数目统计
        rel_tail[r, t] += 1 # support中相同r和t的事实数目统计

    return rel_head, rel_tail

def split_by_time3(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t:
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    return snapshot_list

def get_entity_related_relation(snap, num_nodes, num_rels):
    weight, row, col = [],[],[]
    d = defaultdict(list)
    for triple in snap:
        d[triple[0]].append(triple[1])
    for i in range(num_nodes):
        for j in d[i]:
            weight.append(1/len(d[i]))
            row.append(i)
            col.append(j)
    return sp.csr_matrix((weight, (row, col)), shape=(num_nodes, num_rels*2))

def get_inverse(snap_list, num_rel):
    all_list = []
    for triples in snap_list:
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rel
        all_list.append(inverse_triples)
    return all_list