import tensorflow as tf

from common.io_util import read_training_data_pair, get_batch_data_memory
from tf_train.deep_model.base_nn import BaseModel
from tf_train.deep_model.common_nn import build_pair_embedding_layer, build_pair_full_layer, \
    build_pair_classification_layer, build_pair_regression_layer


class DeepPairwiseModel(BaseModel):
    def __init__(self, _config):
        self.emb_weight = None
        super(DeepPairwiseModel, self).__init__(_config)

    def _input_initializer(self):
        # Inputs.
        with tf.name_scope('inputs'):
            self.pos = tf.placeholder(
                tf.float32, [None, self.config['feature_num']], 'positive')
            self.neg = tf.placeholder(
                tf.float32, [None, self.config['feature_num']], 'negative')
            self.label_pos = tf.placeholder(
                tf.float32, [None, self.config['class_num']], 'label_positive')
            self.label_neg = tf.placeholder(
                tf.float32, [None, self.config['class_num']], 'label_negative')

    def _data_processor(self):
        self.train_pos, self.train_neg, self.y_pos, self.y_neg = \
            read_training_data_pair(self.config['training'],
                                    self.config['feature_num'],
                                    self.config['class_num'])

    def batch_train(self, session, batch_size, begin):
        batch_x_pos, batch_y_pos = get_batch_data_memory(self.train_pos, self.y_pos, batch_size, begin)
        batch_x_neg, batch_y_neg = get_batch_data_memory(self.train_neg, self.y_neg, batch_size, begin)
        _feed_dict = {self.pos: batch_x_pos, self.label_pos: batch_y_pos, self.neg: batch_x_neg,
                      self.label_neg: batch_y_neg}
        _, self.batch_cost, self.batch_accuracy = session.run([self.optimizer, self.cost, self.accuracy],
                                                              feed_dict=_feed_dict)

    def build_net(self):
        # Embedding Layer
        trans_fea_pos, trans_fea_neg, self.emb_weight = build_pair_embedding_layer(self.pos, self.neg, self.config[
            'feature_num'], self.config['embedding_num'])

        # Fully Connected Layer
        for full_layer_num in range(len(self.config['full_output_num'])):
            name_scope = 'Full_Layer_' + str(full_layer_num)
            if full_layer_num == 0:
                fc_pos, fc_neg = build_pair_full_layer(trans_fea_pos,
                                                       trans_fea_neg,
                                                       self.config['embedding_num'],
                                                       self.config['full_output_num'][full_layer_num],
                                                       self.config['drop_out'],
                                                       self.config['activation'],
                                                       name_scope)
            else:
                fc_pos, fc_neg = build_pair_full_layer(fc_pos,
                                                       fc_neg,
                                                       self.config['full_output_num'][
                                                           full_layer_num - 1],
                                                       self.config['full_output_num'][full_layer_num],
                                                       self.config['drop_out'],
                                                       self.config['activation'],
                                                       name_scope)
        class_pos, class_neg = build_pair_classification_layer(fc_pos, fc_neg, self.config['full_output_num'][-1],
                                                               self.config['class_num'])
        reg_pos, reg_neg = build_pair_regression_layer(fc_pos, fc_neg, self.config['full_output_num'][-1],
                                                       self.config['activation'])
        return reg_pos, reg_neg, class_pos, class_neg

    def _eval_cost(self):
        # rank loss
        self.diff = self.predictor[1] - self.predictor[0]
        self.accuracy = tf.greater(self.predictor[0], self.predictor[1])
        self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))
        assert('epsilon' in self.config or 'sigma' in self.config)
        if 'epsilon' in self.config:
            self.cost = tf.reduce_mean(tf.maximum(self.diff + self.config['epsilon'], 0))
        elif 'sigma' in self.config:
            self.cost = tf.reduce_mean(tf.log(1 + tf.exp(self.config['sigma'] * self.diff)))
        # classification loss
        if 'alpha' in self.config and self.label_pos is not None and self.label_neg is not None:
            diff1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictor[2], labels=self.label_pos)
            diff2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictor[3], labels=self.label_neg)
            diff = tf.reduce_mean(diff1) + tf.reduce_mean(diff2)
            self.cost = self.cost * self.config['alpha'] + diff * (1 - self.config['alpha'])
        if 'alpha_sim' in self.config:
            # emb_weight_similarity penalty
            sim_penalty = tf.zeros(dtype=tf.float32, shape=[])
            emb_weight_l2_norm = tf.transpose((tf.nn.l2_normalize(self.emb_weight, axis=0)))
            for i in range(self.config['embedding_num']):
                for j in range(i + 1, self.config['embedding_num']):
                    new_loss = 1 - tf.losses.cosine_distance(tf.nn.embedding_lookup(emb_weight_l2_norm, i),
                                                             tf.nn.embedding_lookup(emb_weight_l2_norm, j), axis=0)
                    sim_penalty += new_loss
            self.cost = self.cost * self.config['alpha_sim'] + sim_penalty * (1 - self.config['alpha_sim'])

    def _build_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.config[
            'learning_rate']).minimize(self.cost)
