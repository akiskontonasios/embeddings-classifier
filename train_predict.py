import sys
import logging

from embeddings.neural_net_architecture import nn_artifact
from embeddings.utils import Action, create_parser, convert_to_binary
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


class LoadDataAction(Action):
    @property
    def name(self):
        return 'Load training/validation/testing data'

    def prepare_args(self, parser):
        parser.add_argument('-train', '--training_data', dest='training_data',
                            help='Imports training data',
                            metavar='training_data', required=True)

        parser.add_argument('-testing', '--testing_data', dest='testing_data',
                            help='Imports testing data',
                            metavar='testing_data', required=True)

        parser.add_argument('-validation', '--validation_data', dest='validation_data',
                            help='Imports validation data',
                            metavar='validation_data', required=True)

    def run(self, args, context=None):

        context['training_data'], context['training_labels'] = self.read_data(args.training_data)
        context['testing_data'], context['testing_labels'] = self.read_data(args.testing_data)
        context['validation_data'], context['validation_labels'] = self.read_data(args.validation_data)

    def read_data(self, path):

        input_data = np.genfromtxt(path, dtype='float', delimiter=',')
        labels = input_data[:, 100]
        input_data = np.delete(input_data, [100, 101, 102], 1)

        return input_data, labels


class TrainAndPredictAction(Action):
    @property
    def name(self):
        return 'Train NN'

    def run(self, args, context=None):

        nn = nn_artifact()

        train_size = context['training_data'].shape[0]
        validation_size = context['validation_data'].shape[0]

        nn.train_nn(train_size + validation_size, context['training_data'], context['training_labels'], context['validation_data'], context['validation_labels'])
        predicted_labels = nn.predict_using_nn(context['testing_data'])

        binary_real_labels = convert_to_binary(context['testing_labels'])
        performance = precision_recall_fscore_support(binary_real_labels, predicted_labels.tolist(), average='binary')
        print(performance)


def main(argv=None):
    program_shortdesc = __import__('__main__').__doc__
    actions = [
               LoadDataAction(),
               TrainAndPredictAction()]
    parser = create_parser(actions, program_shortdesc)
    args = parser.parse_args(argv)
    context = {}
    for action in actions:
        action.run(args, context)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.warning('keyboard interrupt')