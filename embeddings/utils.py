from argparse import ArgumentParser, RawDescriptionHelpFormatter
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Action(object):
    """
    Base action for command line utilities. Each command line utility may consist of one or more actions.
    """

    def prepare_args(self, parser):
        """
        Add arguments to argument parser.
        :param parser: the argument parser to add arguments to.
        """
        pass

    @property
    def name(self):
        """
        Returns the name of the action.
        :return: the name of the action.
        """
        raise ValueError('You need to define a name for the action.')

    @property
    def skipable(self):
        """
        Whether this action can be skipped or not. Default true.
        :return: Whether this action can be skipped or not. Default true.
        """
        return True

    def run(self, args, context=None):
        """
        Runs the actual logic of the action.
        :param args: command line arguments.
        :param context: state passed between actions.
        """
        pass


def create_parser(actions, description):

    parser = ArgumentParser(
        description=description,
        formatter_class=RawDescriptionHelpFormatter)

    for action in actions:
        action.prepare_args(parser)

    return parser


def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors"""

    labels_dense_binary = [1 if label == 'yes' else 0 for label in labels_dense]
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense_binary] = 1

    return labels_one_hot


def convert_to_binary(labels):

    return [1 if label == 'yes' else 0 for label in labels]


def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()

    return temp_batch



