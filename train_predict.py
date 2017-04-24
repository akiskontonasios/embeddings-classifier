import sys
import logging

from embeddings.neural_net_architecture import NNArtifact
from embeddings.utils import Action, create_parser, convert_to_binary
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


class LoadDataAction(Action):
    @property
    def name(self):
        return 'Load data'

    def prepare_args(self, parser):
        parser.add_argument('-d', '--data', dest='data',
                            help='Imports embeddings',
                            metavar='data', required=True)

    def run(self, args, context=None):

        context['data'] = np.genfromtxt(args.data, dtype='float', delimiter=',')


class LoadLabelsAction(Action):
    @property
    def name(self):
        return 'Load labels'

    def prepare_args(self, parser):
        parser.add_argument('-l', '--labels', dest='labels',
                            help='Loads labels',
                            metavar='labels', required=True)

    def run(self, args, context=None):

        context['labels'] = np.genfromtxt(args.labels, dtype='str')


class TrainAndPredictAction(Action):
    @property
    def name(self):
        return 'Train NN'

    def run(self, args, context=None):

        nn = NNArtifact()

        dataset_size = context['data'].shape[0]
        train_size = int(dataset_size*0.5)
        val_size = int(dataset_size*0.7)

        train_x, val_x, test_x = context['data'][:train_size], context['data'][train_size: val_size], context['data'][val_size:]
        train_y, val_y, test_y = context['labels'][:train_size], context['labels'][train_size: val_size], context['labels'][val_size:]

        nn.train_nn(train_size + val_size, train_x, train_y, val_x, val_y)
        predicted_labels = nn.predict_using_nn(test_x)

        binary_real_labels = convert_to_binary(test_y)
        performance = precision_recall_fscore_support(binary_real_labels, predicted_labels.tolist(), average='binary')
        print(performance)

def main(argv=None):
    program_shortdesc = __import__('__main__').__doc__
    actions = [
               LoadDataAction(),
               LoadLabelsAction(),
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