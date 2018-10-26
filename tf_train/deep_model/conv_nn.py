import tensorflow as tf

from common.io_util import read_svm_data, one_hot_encode, get_batch_data_memory
from tf_train.deep_model.base_nn import BaseModel
from tf_train.deep_model.common_nn import build_conv_layer, build_layer, build_pooling_layer


class ConvolutionModel(BaseModel):
    def __init__(self, _config):
        super(ConvolutionModel, self).__init__(_config)

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

    def _input_initializer(self):
        with tf.name_scope('inputs'):
            self.input = tf.placeholder(
                tf.float32, [None, self.config['feature_num']], 'data')
            # self.input = tf.reshape(self.input, shape=[-1, self.config['feature_num'], self.config['embedd'], 1])
            self.label = tf.placeholder(tf.float32, [None, self.config['class_num']], 'label')

    def batch_train(self, session, batch_size, begin):
        batch_x, batch_y = get_batch_data_memory(self.train_x, self.train_y, batch_size, begin)
        _, self.batch_cost, self.batch_accuracy = session.run([self.optimizer, self.cost, self.diff],
                                                              feed_dict={self.input: batch_x, self.label: batch_y})

    def _data_processor(self):
        self.train_x, self.train_y = read_svm_data(self.config['training'], self.config['feature_num'])
        self.test_x, self.test_y = read_svm_data(self.config['test'], self.config['feature_num'])
        self.train_y = one_hot_encode(self.train_y)

    def build_net(self):
        # Reshape input
        # original shape 2D tensor 1 * feature_dim
        # reshape into 3D tensor
        # shape = [batch_size, 1, in_width (feature_dim), in_channel (pointwise 1, pairwise 2)]
        self.input = tf.reshape(self.input, shape=[-1, self.config['feature_num'], 1])
        assert (len(self.config['conv_size']) == len(self.config['filter_size']))
        assert (len(self.config['conv_size']) == len(self.config['pooling_size']))
        conv = self.input
        for layer_num in range(len(self.config['conv_size'])):
            name_scope = 'Conv_Layer_' + str(layer_num)
            with tf.name_scope(name_scope):
                conv = build_conv_layer(conv, self.config['filter_size'][layer_num], 1, self.config[
                    'conv_size'][layer_num], 'Conv_' + str(layer_num), stride=self.config['stride_size'], seed=1234567)
            name_scope = 'Pooling_Layer' + str(layer_num)
            with tf.name_scope(name_scope):
                conv = build_pooling_layer(conv, k=self.config['pooling_size'][layer_num])

        # Fully Connected Layer
        # Reshape conv2 output to fit fully connected layer input
        # Fully Connected Layer Input Number
        fc = conv
        for layer_num in range(len(self.config['full_output_num'])):
            name_scope = "Full_Layer_" + layer_num
            with tf.name_scope(name_scope):
                fc = build_layer(fc, self.config['full_output_num'][layer_num], "full", 1234567)
                fc = tf.nn.relu(fc)
                # Apply Dropout
                fc = tf.nn.dropout(fc, self.config['drop_out'])

        # Output Layer
        with tf.name_scope('Output_Layer'):
            out = build_layer(fc, self.config['class_num'], 'output', 1234567)
        return out
