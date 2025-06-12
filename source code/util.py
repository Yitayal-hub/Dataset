from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import math
import shutil

import numpy as np
import six
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pyhocon
from argparse import ArgumentParser
from colorama import Back, Style
from pip._vendor.colorama import Fore


def initialize_from_env(experiment, logdir=None):

    print("Running experiment: {}".format(experiment))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[experiment]

    if logdir is None:
        logdir = experiment

    config["log_dir"] = mkdirs(os.path.join(config["log_root"], logdir))

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def get_args():
    parser = ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('-l', '--logdir')
    parser.add_argument('--latest-checkpoint', action='store_true')
    return parser.parse_args()


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)


def make_summary(value_dict):
    return tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def flatten(l):
    return [item for sublist in l for item in sublist]


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=None):
    return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def highway(inputs, num_layers, dropout):
    for i in range(num_layers):
        with tf.compat.v1.variable_scope("highway_{}".format(i)):
            j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
            f = tf.sigmoid(f)
            j = tf.nn.relu(j)
            if dropout is not None:
                j = tf.nn.dropout(j, rate=1 - (dropout))
            inputs = f * j + (1 - f) * inputs
    return inputs


def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(input=x)[dim]


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

    if len(inputs.get_shape()) == 3:
        batch_size = shape(inputs, 0)
        seqlen = shape(inputs, 1)
        emb_size = shape(inputs, 2)
        current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.compat.v1.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size],
                                         initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        hidden_bias = tf.compat.v1.get_variable("hidden_bias_{}".format(i), [hidden_size],
                                      initializer=tf.compat.v1.zeros_initializer())
        current_outputs = tf.nn.relu(tf.compat.v1.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, rate=1 - (dropout))
        current_inputs = current_outputs

    output_weights = tf.compat.v1.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.compat.v1.get_variable("output_bias", [output_size],
                                  initializer=tf.compat.v1.zeros_initializer())
    outputs = tf.compat.v1.nn.xw_plus_b(current_inputs, output_weights, output_bias)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
    return outputs

def batch_gather(emb, indices):
    batch_size = shape(emb, 0)
    seqlen = shape(emb, 1)
    if len(emb.get_shape()) > 2:
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
    offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
    gathered = tf.gather(flattened_emb, indices + offset)  # [batch_size, num_indices, emb]
    if len(emb.get_shape()) == 2:
        gathered = tf.squeeze(gathered, 2)  # [batch_size, num_indices]
    return gathered


def assert_rank(tensor, expected_rank, name=None):
    
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.compat.v1.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, rate=1 - (1.0 - dropout_prob))
    return output



FORES = [Fore.BLUE,
         Fore.CYAN,
         Fore.GREEN,
         Fore.MAGENTA,
         Fore.RED,
         Fore.YELLOW]
BACKS = [Back.BLUE,
         Back.CYAN,
         Back.GREEN,
         Back.MAGENTA,
         Back.RED,
         Back.YELLOW]
COLOR_WHEEL = FORES + [f + b for f in FORES for b in BACKS]


def anapho_pprint(tokens, clusters):
    clusters = [tuple(tuple(m) for m in c) for c in clusters]
    cluster_to_color = {c: i % len(COLOR_WHEEL) for i, c in enumerate(clusters)}
    pretty_str = ''
    color_stack = []
    for i, t in enumerate(tokens):
        for c in clusters:
            for start, end in sorted(c, key=lambda m: m[1]):
                if i == start:
                    cluster_color = cluster_to_color[c]
                    pretty_str += Style.BRIGHT + COLOR_WHEEL[cluster_color]
                    color_stack.append(cluster_color)

        pretty_str += t + u' '

        for c in clusters:
            for start, end in c:
                if i == end:
                    pretty_str += Style.RESET_ALL
                    color_stack.pop(-1)
                    if color_stack:
                        pretty_str += Style.BRIGHT + COLOR_WHEEL[color_stack[-1]]

    print(pretty_str)

class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1


class EmbeddingDictionary(object):
    def __init__(self, info, normalize=True, maybe_cache=None):
        self._size = info["size"]
        self._normalize = normalize
        self._path = info["path"]
        if maybe_cache is not None and maybe_cache._path == self._path:
            assert self._size == maybe_cache._size
            self._embeddings = maybe_cache._embeddings
        else:
            self._embeddings = self.load_embedding_dict(self._path)

    @property
    def size(self):
        return self._size

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = np.zeros(self.size)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path) as f:
                for i, line in enumerate(f.readlines()):
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
                    if i == 0 and len(embedding) == 1:
                      continue
                    assert len(embedding) == self.size, "%d,%d: %d: %s" % (len(embedding), self.size, i, line)
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict

    def __getitem__(self, key):
        embedding = self._embeddings[key]
        if self._normalize:
            embedding = self.normalize(embedding)
        return embedding

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return v


class CustomLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, batch_size, dropout):
        self._num_units = num_units
        self._dropout = dropout
        self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), rate=1 - (dropout))
        self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
        initial_cell_state = tf.compat.v1.get_variable("lstm_initial_cell_state", [1, self.output_size])
        initial_hidden_state = tf.compat.v1.get_variable("lstm_initial_hidden_state", [1, self.output_size])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_cell_state, initial_hidden_state)

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self.output_size, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def initial_state(self):
        return self._initial_state

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.compat.v1.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
            c, h = state
            h *= self._dropout_mask
            concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
            i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
            i = tf.sigmoid(i)
            new_c = (1 - i) * c + i * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

    def _orthonormal_initializer(self, scale=1.0):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
            M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return params

        return _initializer

    def _block_orthonormal_initializer(self, output_sizes):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            assert len(shape) == 2
            assert sum(output_sizes) == shape[1]
            initializer = self._orthonormal_initializer()
            params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
            return params

        return _initializer


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def compute_p_m_entity(p_m_link, k):
    p_m_entity = tf.concat([[[1.]], tf.zeros([1, k - 1])], 1)

    def _time_step(i, p_m_entity):
        p_m_e = p_m_entity[:, :i]  # [i, i]  x[i, j] = p(m_i \in E_j)
        p_m_link_i = p_m_link[i:i + 1, :i]  # [1, i]  x[0, j] = p(a_i = j)
        p_m_e_i = tf.matmul(p_m_link_i, p_m_e)  # [1, i]  x[0, j] = \sum_k (p(a_i = k) * p(m_k \in E_j))
        p_m_e_i = tf.concat([p_m_e_i, p_m_link[i:i + 1, i:i + 1]], 1)
        p_m_e_i = tf.pad(tensor=p_m_e_i, paddings=[[0, 0], [0, k - i - 1]], mode='CONSTANT')
        p_m_entity = tf.concat([p_m_entity, p_m_e_i], 0)
        return i + 1, p_m_entity

    _, p_m_entity = tf.while_loop(cond=lambda i, *_: tf.less(i, k),
                                  body=_time_step,
                                  loop_vars=(tf.constant(1), p_m_entity),
                                  shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, None])))

    return p_m_entity

def compute_b3_lost(p_m_entity, x_gold_class_cluster_ids_supgen, k, beta=2.0):
    # remove singleton entities
    gold_entities = tf.reduce_sum(input_tensor=x_gold_class_cluster_ids_supgen, axis=0) > 1.2

    sys_m_e = tf.one_hot(tf.argmax(input=p_m_entity, axis=1), k)
    sys_entities = tf.reduce_sum(input_tensor=sys_m_e, axis=0) > 1.2

    gold_entity_filter = tf.reshape(tf.compat.v1.where(gold_entities), [-1])
    gold_cluster = tf.gather(tf.transpose(a=x_gold_class_cluster_ids_supgen), gold_entity_filter)

    sys_entity_filter, merge = tf.cond(pred=tf.reduce_any(input_tensor=sys_entities & gold_entities),
                                       true_fn=lambda: (tf.reshape(tf.compat.v1.where(sys_entities), [-1]), tf.constant(0)),
                                       false_fn=lambda: (
                                       tf.reshape(tf.compat.v1.where(sys_entities | gold_entities), [-1]), tf.constant(1)))
    system_cluster = tf.gather(tf.transpose(a=p_m_entity), sys_entity_filter)

    # compute intersections
    gold_sys_intersect = tf.pow(tf.matmul(gold_cluster, system_cluster, transpose_b=True), 2)
    r_num = tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=gold_sys_intersect, axis=1) / tf.reduce_sum(input_tensor=gold_cluster, axis=1))
    r_den = tf.reduce_sum(input_tensor=gold_cluster)
    recall = tf.reshape(r_num / r_den, [])

    sys_gold_intersection = tf.transpose(a=gold_sys_intersect)
    p_num = tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=sys_gold_intersection, axis=1) / tf.reduce_sum(input_tensor=system_cluster, axis=1))
    p_den = tf.reduce_sum(input_tensor=system_cluster)
    prec = tf.reshape(p_num / p_den, [])

    beta_2 = beta ** 2
    f_beta = (1 + beta_2) * prec * recall / (beta_2 * prec + recall)

    lost = -f_beta
    # lost = tf.Print(lost, [merge,
    #                        r_num, r_den, p_num, p_den,
    #                        gold_entity_filter, sys_entity_filter,  # tf.reduce_sum(p_m_entity, 0),
    #                        beta, recall, prec, f_beta], summarize=1000)

    return tf.cond(pred=tf.reduce_all(input_tensor=[r_num > .1, p_num > .1, r_den > .1, p_den > .1]),
                   true_fn=lambda: lost,
                   false_fn=lambda: tf.stop_gradient(tf.constant(0.)))
