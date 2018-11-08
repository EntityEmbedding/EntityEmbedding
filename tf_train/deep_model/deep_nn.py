import tensorflow as tf

from common.io_util import read_svm_data, one_hot_encode, get_batch_data_memory
from tf_train.deep_model.base_nn import BaseModel
from tf_train.deep_model.common_nn import build_layer


class DeepModel(BaseModel):
    def __init__(self, _config):
        super(DeepModel, self).__init__(_config)

    def _eval_cost(self):
        diff = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.predictor, labels=self.label)
        self.cost = tf.reduce_mean(diff)
        correct_pred = tf.equal(
            tf.argmax(self.predictor, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def _input_initializer(self):
        # Inputs.
        with tf.name_scope('inputs'):
            self.input = tf.placeholder(
                tf.float32, [None, self.config['feature_num']], 'data')
            self.label = tf.placeholder(
                tf.float32, [None, self.config['class_num']], 'label')

    def _data_processor(self):
        self.train_x, self.train_y = read_svm_data(self.config['training'], self.config['feature_num'])
        self.test_x, self.test_y = read_svm_data(self.config['test'], self.config['feature_num'])
        self.train_y = one_hot_encode(self.train_y, self.config['class_num'])
        self.test_y = one_hot_encode(self.test_y, self.config['class_num'])

    def batch_train(self, session, batch_size, begin):
        batch_x, batch_y = get_batch_data_memory(self.train_x, self.train_y, batch_size, begin)
        _, self.batch_cost, self.batch_accuracy = session.run([self.optimizer, self.cost, self.accuracy],
                                                              feed_dict={self.input: batch_x, self.label: batch_y})

    def _build_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.config[
            'learning_rate']).minimize(self.cost)

    def build_net(self):
        # Embedding Layer
        with tf.name_scope('Embedding_Layer'):
            trans_fea = build_layer(self.input, self.config['embedding_num'], 'embedding')

        # Full Layer
        with tf.name_scope('Full_Layer'):
            fc = build_layer(trans_fea, self.config['full_output_num'], 'full')
            fc = tf.nn.relu(fc)
            # Apply Dropout
            fc = tf.nn.dropout(fc, self.config['drop_out'])

        # Output Layer
        with tf.name_scope('Output_Layer'):
            out = build_layer(fc, self.config['class_num'], 'output')
        return out
