import numpy as np
import math
from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory

mem = Memory("./mycache")
ZERO = 0.00000000001


@mem.cache
def read_svm_data(train_data_file, feature_num):
    print(train_data_file)
    data = load_svmlight_file(train_data_file, zero_based=True, n_features=feature_num)
    features = data[0].toarray()
    labels = data[1]
    return features, labels


def one_hot_encode(labels, label_num=None):
    labels = labels.astype(int)
    n_labels = len(labels)
    if label_num is None:
        n_unique_labels = len(np.unique(labels))
        _one_hot_encode = np.zeros((n_labels, n_unique_labels))
    else:
        _one_hot_encode = np.zeros((n_labels, label_num))
    _one_hot_encode[np.arange(n_labels), labels] = 1
    return _one_hot_encode


def get_batch_data_memory(x, y, batch_size, begin):
    batch_data = np.zeros((batch_size, x.shape[1]))
    tmp_data = x[begin:begin + batch_size, :]
    if type(tmp_data) is not np.ndarray:
        tmp_data = tmp_data.toarray()  # convert sparse matrices
    batch_data[:batch_size] = tmp_data
    batch_labels = None
    if y is not None:
        batch_labels = np.zeros((batch_size, y.shape[1]))
        batch_labels[:batch_size] = y[begin:begin + batch_size]
    return batch_data, batch_labels


@mem.cache
def read_training_data_pair(train_data_file, feature_num, class_num):
    data = load_svmlight_file(train_data_file, zero_based=True, n_features=feature_num)
    features = data[0]
    labels = data[1]
    features = features.toarray()
    train_pos = []
    train_neg = []
    train_pos_y = []
    train_neg_y = []
    for i in range(len(labels)):
        if i % 2 == 0:
            train_pos.append(features[i])
            train_pos_y.append(labels[i])
        else:
            train_neg.append(features[i])
            train_neg_y.append(labels[i])
    train_pos = np.array(train_pos)
    train_pos_y = np.array(train_pos_y)
    train_neg = np.array(train_neg)
    train_neg_y = np.array(train_neg_y)
    train_pos_y = one_hot_encode(train_pos_y, class_num)
    train_neg_y = one_hot_encode(train_neg_y, class_num)
    return train_pos, train_neg, train_pos_y, train_neg_y


def read_group(group_file):
    group = np.loadtxt(group_file)
    group = group.astype(int)
    return group


def _convert(t):
    """Convert feature and value to appropriate types"""
    return int(t[0]), float(t[1])


def parse_svmlight_line(line, feature_num):
    data = line.split()
    label = float(data[0])
    feature_vec = [np.NAN] * feature_num
    features = [_convert(feature.split(':')) for feature in data[1:]]
    if features and max([t[0] for t in features]) + 1 > feature_num:
        raise Exception("data don't match feature config")
    for t in features:
        feature_vec[t[0]] = t[1]
    return label, feature_vec


def parse_svmlight_label(line):
    data = line.strip().split(' ', 1)
    if len(data) == 2:
        label = data[0]
        features = data[1]
    else:
        label = data[0]
        features = ''
    return label


def padding_to_vec(padding_lines, feature_num, seq_num):
    dim = feature_num * seq_num
    vec = [0] * dim
    idx = 0
    for line in padding_lines:
        _, feat_vec = parse_svmlight_line(line, feature_num)
        for i in range(len(feat_vec)):
            value = float(feat_vec[i])
            if is_not_nan_value(value):
                vec[idx] = value
            idx += 1
    return vec


def is_not_nan_value(num):
    if not math.isnan(num) and abs(num) > ZERO and not math.isinf(num):
        return True
    else:
        return False


def read_training_sequence(train_data_file, feature_num, seq_num):
    group_file = train_data_file + '.group'
    group = read_group(group_file)
    padding = []
    label_avg = 0.0
    result = []
    label_new = []
    with open(train_data_file, buffering=(2 << 16) + 8) as f_in:
        for g_size in group:
            session = []
            for g in range(g_size):
                for line in f_in:
                    label = parse_svmlight_label(line)
                    session.append((label, line))
                    break
            session_sorted = sorted(session, key=lambda x: x[0])
            for i in range(len(session)):
                padding.append(session_sorted[i][1])
                label_avg += float(session_sorted[i][0])
                if i % seq_num == 0 and len(padding) > 0:
                    vec = padding_to_vec(padding, feature_num, seq_num)
                    result.append(vec)
                    label_new.append(label_avg / len(padding))
                    padding = []
                    label_avg = 0.0
            if len(padding) > 0:
                # adding the remaining into result
                vec = padding_to_vec(padding, feature_num, seq_num)
                result.append(vec)
                label_new.append(int(label_avg / len(padding)))
                padding = []
                label_avg = 0.0
    return np.asarray(result), np.asarray(label_new)
