import argparse
import os
import pickle

import numpy as np
import tqdm
from scipy.sparse import csc_matrix

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="ICEWS14")
args = parser.parse_args()

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

num_e, num_r = get_total_number(args.dataset, "stat.txt")

def load_quadruples(inPath, fileName):
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
    return np.asarray(quadrupleList)

def get_his_frequency(quadruples, num_e, num_r, s_history, o_history, r_history):
    rr = quadruples[:, 1]  # 关系
    s_frequency = np.zeros((quadruples.shape[0], num_e), dtype=float)
    o_frequency = np.zeros((quadruples.shape[0], num_e), dtype=float)
    r_frequency = np.zeros((quadruples.shape[0], num_r), dtype=float)

    for ix in tqdm.tqdm(range(quadruples.shape[0])):
        for con_events in s_history[ix]:
            idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
            cur_events = con_events[idxx, 1].tolist()
            s_frequency[ix][cur_events] += 1

        for con_events in o_history[ix]:
            idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
            cur_events = con_events[idxx, 1].tolist()
            o_frequency[ix][cur_events] += 1

        for con_events in r_history[ix]:
            idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
            cur_events = con_events[idxx, 1].tolist()
            r_frequency[ix][cur_events] += 1

    s_frequency = csc_matrix(s_frequency)
    o_frequency = csc_matrix(o_frequency)
    r_frequency = csc_matrix(r_frequency)

    return s_frequency, o_frequency, r_frequency

s_his = [[] for _ in range(num_e)]
o_his = [[] for _ in range(num_e)]
r_his = [[] for _ in range(num_r)]
s_his_cache = [[] for _ in range(num_e)]
o_his_cache = [[] for _ in range(num_e)]
r_his_cache = [[] for _ in range(num_r)]
latest_t = 0

def get_frequency(data, num_e, num_r, mode="online"):
    global s_his, o_his, r_his, s_his_cache, o_his_cache, r_his_cache, latest_t

    s_history_data = [[] for _ in range(len(data))]
    o_history_data = [[] for _ in range(len(data))]
    r_history_data = [[] for _ in range(len(data))]

    for i, tuples in enumerate(data):
        t = tuples[3]  # 时间戳
        if latest_t != t:
            for ee in range(num_e):
                if len(s_his_cache[ee]) != 0:
                    s_his[ee].append(s_his_cache[ee].copy())
                    s_his_cache[ee] = []
                if len(o_his_cache[ee]) != 0:
                    o_his[ee].append(o_his_cache[ee].copy())
                    o_his_cache[ee] = []

            for rr in range(num_r):
                if len(r_his_cache[rr]) != 0:
                    r_his[rr].append(r_his_cache[rr].copy())
                    r_his_cache[rr] = []

            latest_t = t

        s, r, o = tuples[0], tuples[1], tuples[2]
        s_history_data[i] = s_his[s].copy()
        o_history_data[i] = o_his[o].copy()
        r_history_data[i] = r_his[r].copy()

        if mode == "online":
            if len(s_his_cache[s]) == 0:
                s_his_cache[s] = np.array([[r, o]])
            else:
                s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)

            if len(o_his_cache[o]) == 0:
                o_his_cache[o] = np.array([[r, s]])
            else:
                o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)

            if len(r_his_cache[r]) == 0:
                r_his_cache[r] = np.array([[s, o]])
            else:
                r_his_cache[r] = np.concatenate((r_his_cache[r], [[s, o]]), axis=0)

        elif mode == "offline":
            pass
        else:
            print("Invalid mode!")
            exit()

    s_frequency, o_frequency, r_frequency = get_his_frequency(data, num_e, num_r, s_history_data, o_history_data, r_history_data)
    return s_frequency, o_frequency, r_frequency

def get_all_frequency(dataset, num_e, num_r):
    train_data = load_quadruples(dataset, "train.txt")
    if dataset != "ICEWS14":
        dev_data = load_quadruples(dataset, "valid.txt")
    test_data = load_quadruples(dataset, "test.txt")

    print("Start to statistics training set...")
    train_s_frequency, train_o_frequency, train_r_frequency = get_frequency(train_data, num_e, num_r)
    print("The statistics on the training set have been completed.")

    if dataset != "ICEWS14":
        print("Start to statistics validation set...")
        dev_s_frequency, dev_o_frequency, dev_r_frequency = get_frequency(dev_data, num_e, num_r)
        print("The statistics on the validation set have been completed.")

    print("Start to statistics test set...")
    test_s_frequency_offline, test_o_frequency_offline, test_r_frequency_offline = get_frequency(test_data, num_e, num_r, "offline")
    test_s_frequency, test_o_frequency, test_r_frequency = get_frequency(test_data, num_e, num_r)
    print("The statistics on the test set have been completed.")

    print("Write to files...")
    with open(dataset + '/train_s_frequency.txt', 'wb') as fp:
        pickle.dump(train_s_frequency, fp)
    with open(dataset + '/train_o_frequency.txt', 'wb') as fp:
        pickle.dump(train_o_frequency, fp)
    with open(dataset + '/train_r_frequency.txt', 'wb') as fp:
        pickle.dump(train_r_frequency, fp)

    if dataset != "ICEWS14":
        with open(dataset + '/dev_s_frequency.txt', 'wb') as fp:
            pickle.dump(dev_s_frequency, fp)
        with open(dataset + '/dev_o_frequency.txt', 'wb') as fp:
            pickle.dump(dev_o_frequency, fp)
        with open(dataset + '/dev_r_frequency.txt', 'wb') as fp:
            pickle.dump(dev_r_frequency, fp)

    with open(dataset + '/test_s_frequency.txt', 'wb') as fp:
        pickle.dump(test_s_frequency, fp)
    with open(dataset + '/test_o_frequency.txt', 'wb') as fp:
        pickle.dump(test_o_frequency, fp)
    with open(dataset + '/test_r_frequency.txt', 'wb') as fp:
        pickle.dump(test_r_frequency, fp)
    with open(dataset + '/test_s_frequency_offline.txt', 'wb') as fp:
        pickle.dump(test_s_frequency_offline, fp)
    with open(dataset + '/test_o_frequency_offline.txt', 'wb') as fp:
        pickle.dump(test_o_frequency_offline, fp)
    with open(dataset + '/test_r_frequency_offline.txt', 'wb') as fp:
        pickle.dump(test_r_frequency_offline, fp)

if __name__ == '__main__':
    get_all_frequency(args.dataset, num_e, num_r)
