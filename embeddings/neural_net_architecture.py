import tensorflow as tf
import numpy as np
import time
from embeddings.utils import dense_to_one_hot, preproc

class nn_artifact():

    seed = 128

    input_num_units = 100
    hidden_num_units = 1000
    output_num_units = 2

    data = tf.placeholder(tf.float32, [None, input_num_units])
    labels = tf.placeholder(tf.float32, [None, output_num_units])

    epochs = 50
    batch_size = 1000
    learning_rate = 0.0001

    sess = tf.Session()

    weights = {
        'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
    }

    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
    }

    def initialize_net(self):

        self.hidden_layer = tf.add(tf.matmul(self.data, self.weights['hidden']), self.biases['hidden'])
        self.hidden_layer = tf.nn.relu(self.hidden_layer)

        self.output_layer = tf.matmul(self.hidden_layer, self.weights['output']) + self.biases['output']

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_layer, self.labels))

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        return self.hidden_layer, self.output_layer, self.cost, self.optimizer

    def train_nn(self, training_set_size, train_data, train_labels, validation_data, validation_labels):

        hidden_layer, output_layer, cost, optimizer = self.initialize_net()
        init = tf.initialize_all_variables()

        self.sess.run(init)

        for epoch in range(self.epochs):
            avg_cost = 0
            running_time_start = time.time()
            total_batch = int(training_set_size/self.batch_size)
            for i in range(total_batch):
                batch_data, batch_labels = self.batch_creator(self.batch_size, train_data.shape[0], train_data, train_labels)
                _, c = self.sess.run([optimizer, cost], feed_dict={self.data: batch_data, self.labels: batch_labels})

            avg_cost += c / total_batch

            print("Epoch:", (epoch+1), "cost =", "{:.20f}".format(avg_cost), ", Time: ", time.time() - running_time_start)

        pred_temp = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("Validation Accuracy:", accuracy.eval(session=self.sess, feed_dict={self.data: validation_data.reshape(-1, self.input_num_units), self.labels: dense_to_one_hot(validation_labels)}))

        print("\nTraining complete!")

    def predict_using_nn(self, test_data):

        print("\nPrediction")
        predict = tf.argmax(self.output_layer, 1)
        pred = predict.eval(session=self.sess, feed_dict={self.data: test_data.reshape(-1, self.input_num_units)})

        return pred

    def batch_creator(self, batch_size, dataset_length, data, labels):
        """Create batch with random samples and return appropriate format"""
        batch_mask = np.random.RandomState(self.seed).choice(dataset_length, batch_size)

        batch_x = data[[batch_mask]].reshape(-1, self.input_num_units)
        # batch_x = preproc(batch_x)

        if len(labels) > 0:
            batch_y = labels[batch_mask]
            batch_y = dense_to_one_hot(batch_y)

        return batch_x, batch_y