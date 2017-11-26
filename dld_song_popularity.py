from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import math
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Global Variables
# MODEL DIRECTORY, TRAINING/VALIDATION/TEST SPLIT
CSV_FILE_NAME = 'raw_data.csv'
NUM_STEPS = 5000
TOTAL_SONGS = 9905
SONGS_TO_USE = 3000
MODEL_DIR = "/Users/bo/Desktop/Deep Learning Decal/checkpoints_3000_0.00005_0.35_5000"
TRAINING_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TESTING_SPLIT = 0.1

# Possible Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 0.00005
DROPOUT_RATE = 0.35


# Our data is in the following format:
# 9905 * 30 * 1000 Tensor
# (9905 Songs, 1000 Sections/Song, 30 Datapoints/Section)

def cnn_model_fn(features, labels, mode):
	""" Model CNN Function """
	# Input layer
	print(features["x"].shape)
	low_level_features = features["x"][:,0:30000]
	low_level_features = tf.reshape(low_level_features, (BATCH_SIZE, 30, 1000))
	high_level_features = features["x"][:,30000:30005]

	input_layer = tf.reshape(low_level_features, [-1, 1000, 30, 1])
	input_layer_2 = tf.reshape(high_level_features, [-1, 5, 1, 1])
	print("Input Layer: " + str(input_layer.shape))

	# Convolutional Layer 1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[4, 30],
		padding="same",
		activation=tf.nn.relu)
	print("Conv1 Layer: " + str(conv1.shape))

	# Pooling Layer 1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=[2, 1])
	print("Pool1 Layer: " + str(pool1.shape))

	# Convolutional Layer 2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=32,
		kernel_size=[4, 30],
		padding="same",
		activation=tf.nn.relu)
	print("Conv2 Layer: " + str(conv2.shape))

	# Pooling Layer 2
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1], strides=[2, 1])
	print("Pool2 Layer: " + str(pool2.shape))

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 30 * 250 * 32])
	pool2_flat = tf.concat([pool2_flat, high_level_features], 1)
	print("Pool2 Flattened: " + str(pool2_flat.shape))
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	print("Dense layer: " + str(dense.shape))
	dropout = tf.layers.dropout(
		inputs=dense, rate=DROPOUT_RATE, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=5)
	print("Logits layer: " + str(logits.shape))

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	#Calculate Loss (TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	#Configure the Training Op, for TRAIN mode
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluatoin metrics (EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels,
			predictions=predictions["classes"]
			)
	}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def popularity(x):
	if x >= 80:
		return 5
	elif x >= 60:
		return 4
	elif x >= 40:
		return 3
	elif x >= 20:
		return 2
	else:
		return 1

def main(unused_argv):
	# Load the data from our csv file
	print("Loading data ...")
	sample_data = np.genfromtxt(CSV_FILE_NAME,delimiter=',',dtype='float32',skip_footer=TOTAL_SONGS - SONGS_TO_USE)
	num_data_points = sample_data.shape[0]
	print("Num data points: " + str(num_data_points))
	validation_split_start = math.floor(TRAINING_SPLIT * num_data_points)
	testing_split_start = math.floor((TRAINING_SPLIT + VALIDATION_SPLIT) * num_data_points)

	popularity_mapper = np.vectorize(popularity)
	training_data = sample_data[:validation_split_start,0:30005]
	training_labels = sample_data[:validation_split_start,30005]
	training_labels = popularity_mapper(training_labels)
	popularity_mapper = np.vectorize(popularity)
	validation_data = sample_data[validation_split_start:testing_split_start,0:30005]
	validation_labels = sample_data[validation_split_start:testing_split_start,30005]
	popularity_mapper = np.vectorize(popularity)
	validation_labels = popularity_mapper(validation_labels)
	testing_data = sample_data[testing_split_start:,0:30005]
	testing_labels = sample_data[testing_split_start:,30005]
	print(testing_labels)
	testing_labels = popularity_mapper(testing_labels)
	print(testing_labels)
	
	# Create Estimator
	classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn,
		model_dir=MODEL_DIR)

	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=5)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": training_data},
		y=training_labels,
		batch_size=BATCH_SIZE,
		num_epochs=None,
		shuffle=True)
	classifier.train(
		input_fn=train_input_fn,
		steps=NUM_STEPS,
		hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": training_data},
		y=training_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == "__main__":
  tf.app.run()