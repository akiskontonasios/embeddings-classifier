import tensorflow as tf
import numpy as np
from embeddings.utils import dense_to_one_hot, preproc


class NNArtifact():

    seed = 128

    input_num_units = 100
    hidden_num_units = 1000
    output_num_units = 2

    x = tf.placeholder(tf.float32, [None, input_num_units])
    y = tf.placeholder(tf.float32, [None, output_num_units])

    epochs = 20
    batch_size = 1000
    learning_rate = 0.0001

    weights = {
        'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
    }

    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
    }

    def initialize_net(self):

        hidden_layer = tf.add(tf.matmul(self.x, self.weights['hidden']), self.biases['hidden'])
        hidden_layer = tf.nn.relu(hidden_layer)

        output_layer = tf.matmul(hidden_layer, self.weights['output']) + self.biases['output']

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, self.y))

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(cost)

        return hidden_layer, output_layer, cost, optimizer

    def train_nn(self, training_set_size, train_x, val_x, val_y):

        hidden_layer, output_layer, cost, optimizer = self.initialize_net()
        init = tf.initialize_all_variables()

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(self.epochs):
                avg_cost = 0
                total_batch = int(training_set_size/self.batch_size)
                for i in range(total_batch):
                    batch_x, batch_y = self.batch_creator(self.batch_size, train_x.shape[0], 'train')
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})

                avg_cost += c / total_batch

                print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))

        print("\nTraining complete!")

        pred_temp = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("Validation Accuracy:", accuracy.eval({self.x: val_x.reshape(-1, self.input_num_units), self.y: dense_to_one_hot(val_y)}))

    def predict_using_nn(self,test_x):

        predict = tf.argmax(self.output_layer, 1)
        pred = predict.eval({self.x: test_x.reshape(-1, self.input_num_units)})

        return pred

    def batch_creator(self, batch_size, dataset_length, dataset_name):
        """Create batch with random samples and return appropriate format"""
        batch_mask = np.random.RandomState(self.seed).choice(dataset_length, batch_size)

        batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, self.input_num_units)
        batch_x = preproc(batch_x)

        if dataset_name == 'train':
            batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
            batch_y = dense_to_one_hot(batch_y)

        return batch_x, batch_y