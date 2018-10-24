import os
import shutil
import time

import numpy as np
import tensorflow as tf


class BaseModel(object):
    def __init__(self, _config):
        self.config = _config
        self.graph = tf.Graph()
        self.predictor = None
        self.optimizer = None
        self.summary = None
        self.saver = None
        self.init = None
        self.batch_cost = np.inf
        self.batch_accuracy = 0
        self.build_graph()

    def build_net(self):
        raise NotImplementedError

    def _eval_cost(self):
        raise NotImplementedError

    def _build_optimizer(self):
        raise NotImplementedError

    def _input_initializer(self):
        raise NotImplementedError

    def _data_processor(self):
        raise NotImplementedError

    def build_graph(self):
        # Build the computational graph of the model
        with self.graph.as_default():
            self._input_initializer()

            # construct the model
            self.predictor = self.build_net()
            self._eval_cost()
            self.optimizer = self._build_optimizer()
            # Initialize variables, i.e. weights and biases.
            self.init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.summary = tf.summary.merge_all()
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('accuracy', self.accuracy)
            self.saver = tf.train.Saver(max_to_keep=5)
        self.graph.finalize()

    def _get_path(self, folder):
        return os.path.join(self.config['output_folder'], folder)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(
                self._get_path('checkpoints'))
            self.saver.restore(sess, filename)
        return sess

    def predict(self, _predictor, _feed_dict, session=None, is_eval=False):
        sess = self._get_session(session)
        loss = 0
        acc = 0
        if is_eval:
            y_pre, loss, acc = sess.run([_predictor, self.cost, self.accuracy], _feed_dict)
        else:
            y_pre = sess.run(_predictor, _feed_dict)
        return y_pre, loss, acc

    def batch_train(self, session, batch_size, begin):
        raise NotImplementedError

    def train_net(self, train_size, is_incremental=False):
        start_time = time.time()
        self._data_processor()
        if train_size is None:
            if hasattr(self,'train_x') and self.train_x is not None:
                train_size = self.train_x.shape[0]
            if hasattr(self,'train_pos') and self.train_pos is not None:
                train_size = self.train_pos.shape[0]
        if is_incremental:
            sess = self._get_session(None)
        else:
            sess = tf.Session(graph=self.graph)
            shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
            os.makedirs(self._get_path('checkpoints'))
            shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        summary_writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        path = os.path.join(self._get_path('checkpoints'), 'model.ckpt')
        sess.run(self.init)
        # Keep training until reach max iterations
        for epoch in range(self.config['training_epochs']):
            batch_size = self.config['batch_size']
            total_batch = int(train_size / batch_size)
            avg_loss = 0
            avg_acc = 0
            begin = 0
            summary = tf.Summary()
            # Loop over all batches
            for i in range(total_batch):
                self.batch_train(sess, batch_size, begin)
                avg_loss += self.batch_cost / total_batch
                avg_acc += self.batch_accuracy / total_batch
                begin += batch_size
            if epoch % self.config['display_step'] == 0:
                print("Iter " + str(epoch) + ", Minibatch Loss= " + "{:.6f}".format(
                    avg_loss) + ", Training Accuracy= " + "{:.5f}".format(avg_acc))
                summary.value.add(tag='accuracy', simple_value=avg_acc)
                summary.value.add(tag='loss', simple_value=avg_loss)
                summary_writer.add_summary(summary, epoch)
                self.saver.save(sess, path, epoch)
        self.saver.save(sess, path, epoch)
        summary_writer.close()
        print("Optimization Finished!")
        sess.close()
        print("--- model train %s seconds ---" % (time.time() - start_time))
