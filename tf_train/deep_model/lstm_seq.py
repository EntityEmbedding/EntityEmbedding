import tensorflow as tf
from tensorflow.contrib import rnn


from common.io_util import read_training_sequence, one_hot_encode, get_batch_data_memory
from tf_train.deep_model.base_nn import BaseModel
from tf_train.deep_model.common_nn import build_layer


class LSTMModel(BaseModel):
    def _input_initializer(self):
        # Inputs.
        with tf.name_scope('inputs'):
            self.input = tf.placeholder(
                tf.float32, [None, self.config['feature_num'], self.config['seq_num']], 'data')
            self.label = tf.placeholder(
                tf.float32, [None, self.config['class_num']], 'label')

    def _data_processor(self):
        self.sequence_num = self.config['seq_num']
        self.train_x, self.train_y = read_training_sequence(self.config['training'],
                                                            self.config['feature_num'],
                                                            self.sequence_num)
        self.test_x, self.test_y = read_training_sequence(self.config['training'],
                                                          self.config['feature_num'],
                                                          self.sequence_num)
        self.test_x = self.test_x.reshape((-1, self.config['feature_num'], self.config['seq_num']))
        self.train_y = one_hot_encode(self.train_y, self.config['class_num'])
        self.test_y = one_hot_encode(self.test_y, self.config['class_num'])

    def batch_train(self, session, batch_size, begin):
        batch_x, batch_y = get_batch_data_memory(self.train_x, self.train_y, batch_size, begin)
        batch_x = batch_x.reshape((batch_size, self.config['feature_num'], self.config['seq_num']))
        _, self.batch_cost, self.batch_accuracy = session.run([self.optimizer, self.cost, self.accuracy],
                                                              feed_dict={self.input: batch_x, self.label: batch_y})

    def __init__(self, _config):
        super(LSTMModel, self).__init__(_config)

    def build_net(self):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(self.input, [1, 0, 2])
        # Reshaping to (n_steps * batch_size, n_input)
        x = tf.reshape(x, [-1, self.config['feature_num']])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, self.config['seq_num'], 0)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(self.config['hidden_num'], forget_bias=self.config['forget'])

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        with tf.name_scope('Output_Layer'):
            out = build_layer(outputs[-1], self.config['class_num'], 'output')
        return out

    def _build_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.config[
            'learning_rate']).minimize(self.cost)

    def _eval_cost(self):
        diff = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.predictor, labels=self.label)
        self.cost = tf.reduce_mean(diff)
        correct_pred = tf.equal(
            tf.argmax(self.predictor, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

