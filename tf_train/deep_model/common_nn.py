import tensorflow as tf


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(),
                     weighted_decay=0.1):
    """
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param weighted_decay: the decay weight of l2 regularizer
    :param initializer: User Xavier as default.
    :return: The created variable
    """
    regularizer = tf.contrib.layers.l2_regularizer(scale=weighted_decay)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def build_layer(input_layer, output_dim, variable_name, seed=None):
    """
    :param input_layer: 2D tensor
    :param variable_name: the name of variable for tensor board
    :param output_dim: int. the dimension of next layer (or output layer)
    :param seed: the seed of random initialization
    :return: output layer Y = WX + B
    """
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name=variable_name + '_weights', shape=[input_dim, output_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0, seed=seed))
    fc_b = create_variables(name=variable_name + '_bias', shape=[output_dim],
                            initializer=tf.zeros_initializer())
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and relu activation, default strides = 1, and
    # use zero-padding.
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1],
                     padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def build_conv_layer(input_layer, filter_size, embed_dim, output_dim, layer_name, strides=1, seed=None):
    input_dim = input_layer.get_shape().as_list()[-1]
    conv_weight = create_variables(name=layer_name + '_weights', shape=[input_dim * filter_size, embed_dim, 1,
                                   output_dim],
                                   initializer=tf.uniform_unit_scaling_initializer(factor=1.0, seed=seed))
    conv_bias = create_variables(name=layer_name + '_bias', shape=[output_dim],
                                 initializer=tf.zeros_initializer())
    x = tf.nn.conv2d(input_layer, conv_weight, strides=[1, strides, strides, 1],
                     padding='SAME')
    x = tf.nn.bias_add(x, conv_bias)
    return tf.nn.relu(x)


def build_pooling_layer(input_layer, k=2):
    # Max Pooling wrapper, default grid = 1 * 2, with non-overlapping strides =
    # grid size
    return tf.nn.max_pool(input_layer, ksize=[1, 1, k, 1], strides=[1, 1, k, 1],
                          padding='SAME')


def build_pair_embedding_layer(pos, neg, feature_dim, embedding_dim):
    # Embedding Layer
    with tf.name_scope('Embedding_Layer'):
        weight = create_variables(name='embedding_weights',
                                  shape=[feature_dim, embedding_dim],
                                  initializer=tf.initializers.variance_scaling())
        bias = create_variables(name='embedding_bias', shape=[embedding_dim],
                                initializer=tf.zeros_initializer())
        trans_fea_pos = tf.matmul(pos, weight)
        trans_fea_pos = tf.nn.bias_add(trans_fea_pos, bias)
        trans_fea_neg = tf.matmul(neg, weight)
        trans_fea_neg = tf.nn.bias_add(trans_fea_neg, bias)
    return trans_fea_pos, trans_fea_neg, weight


def build_pair_full_layer(trans_fea_pos, trans_fea_neg, in_fea_num, out_fea_num, dropout, activation,
                          name_scope='Full_Layer'):
    # Full Layer
    with tf.name_scope(name_scope):
        weight = create_variables(name=name_scope + '_full_weights',
                                  shape=[in_fea_num, out_fea_num],
                                  initializer=tf.initializers.variance_scaling())
        bias = create_variables(name=name_scope + '_full_bias', shape=[out_fea_num],
                                initializer=tf.zeros_initializer())
        fc_pos = tf.matmul(trans_fea_pos, weight) + bias
        fc_neg = tf.matmul(trans_fea_neg, weight) + bias
        if activation == 'tanh':
            fc_pos = tf.nn.tanh(fc_pos)
            fc_neg = tf.nn.tanh(fc_neg)
        elif activation == 'sigmoid':
            fc_pos = tf.nn.sigmoid(fc_pos)
            fc_neg = tf.nn.sigmoid(fc_neg)
        elif activation == 'relu':
            fc_pos = tf.nn.relu(fc_pos)
            fc_neg = tf.nn.relu(fc_neg)
        else:
            pass
        fc_pos = tf.nn.dropout(fc_pos, dropout)
        fc_neg = tf.nn.dropout(fc_neg, dropout)
    return fc_pos, fc_neg


def build_pair_classification_layer(fc_pos, fc_neg, output_num, class_num):
    with tf.name_scope('Classification_Layer'):
        weight = create_variables(name='classification_weights',
                                  shape=[output_num, class_num],
                                  initializer=tf.initializers.variance_scaling())
        bias = create_variables(name='classification_bias', shape=[class_num],
                                initializer=tf.zeros_initializer())
        class_pos = tf.matmul(fc_pos, weight) + bias
        class_neg = tf.matmul(fc_neg, weight) + bias
    return class_pos, class_neg


def build_pair_regression_layer(fc_pos, fc_neg, output_num, activator):
    # Final Output Layer
    with tf.name_scope('Regression_Layer'):
        weight = create_variables(name='reg_weights',
                                  shape=[output_num, 1],
                                  initializer=tf.initializers.variance_scaling())
        bias = create_variables(name='reg_bias', shape=[1],
                                initializer=tf.zeros_initializer())
        reg_pos = tf.matmul(fc_pos, weight) + bias
        reg_neg = tf.matmul(fc_neg, weight) + bias
        if activator == 'tanh':
            reg_pos = tf.nn.tanh(reg_pos)
            reg_neg = tf.nn.tanh(reg_neg)
        elif activator == 'sigmoid':
            reg_pos = tf.nn.sigmoid(reg_pos)
            reg_neg = tf.nn.sigmoid(reg_neg)
        elif activator == 'relu':
            reg_pos = tf.nn.relu(reg_pos)
            reg_neg = tf.nn.relu(reg_neg)
        else:
            pass
    return reg_pos, reg_neg
