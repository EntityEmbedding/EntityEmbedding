import argparse
import json

import numpy as np
from sklearn.metrics import classification_report

from tf_train.deep_model.conv_nn import ConvolutionModel
from tf_train.deep_model.deep_nn import DeepModel
from tf_train.deep_model.lstm_seq import LSTMModel
from tf_train.deep_model.pairwise_deep_nn import DeepPairwiseModel


def create_parser():
    # parse command line input
    parser = argparse.ArgumentParser(description='Using CNN or DNN to train embedding of human crafted features')
    parser.add_argument('--config', '-c', required=True, help='the config file of offline training')
    parser.add_argument('--incremental', '-i', default='0', required=False, choices=['0', '1'],
                        help='whether it is incremental training or not')
    parser.add_argument('--menu', '-m', default='dnn', required=False, choices=['dnn', 'cnn', 'dnn_pair', 'lstm'],
                        help='whether it is using CNN architecture or DNN architecture')
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    is_incremental = (args.incremental == '1')
    with open(args.config, 'r') as cf:
        _config = json.load(cf)
        if args.menu == 'dnn':
            DNN = DeepModel(_config)
            DNN.train_net(None, is_incremental)
            _feed_dict = {DNN.input: DNN.test_x}
            y_pred, _, _ = DNN.predict(DNN.predictor, _feed_dict, None, is_eval=False)
            predict_label = np.argmax(y_pred, axis=1)
            test_label = np.argmax(DNN.test_y, axis=1)
            print('Model Performance of DNN\n')
            print(classification_report(test_label, predict_label))
        elif args.menu == 'cnn':
            CNN = ConvolutionModel(_config)
            CNN.train_net(None, is_incremental)
            _feed_dict = {CNN.input: CNN.test_x}
            y_pred = CNN.predict(CNN.predictor, _feed_dict, None, is_eval=False)
            predict_label = np.argmax(y_pred, axis=1)
            test_label = np.argmax(CNN.test_y, axis=1)
            print('Model Performance of CNN\n')
            print(classification_report(test_label, predict_label))
        elif args.menu == 'dnn_pair':
            PNN = DeepPairwiseModel(_config)
            PNN.train_net(None, is_incremental)
        else:
            LSTM = LSTMModel(_config)
            LSTM.train_net(None, is_incremental)
            _feed_dict = {LSTM.input: LSTM.test_x}
            y_pred = LSTM.predict(LSTM.predictor, _feed_dict, None, is_eval=False)
            predict_label = np.argmax(y_pred, axis=1)
            test_label = np.argmax(LSTM.test_y, axis=1)
            print('Model Performance of DNN\n')
            print(classification_report(test_label, predict_label))
