from __future__ import absolute_import, division, print_function

import os
import json
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from pprint import pprint
from process_data import get_segments


def get_data_from_json(filename, ratio=30):
    """
    Args:
      filename: the json filename containing the data
      ratio: the retio between the train and tests set devision 
            (10 means the test set will get 10% of the data and the train set wil get 90%)
    Returns:
      List of (train_features, train_labels, test_features, test_labels, vocabulary)
    """

    #load dataset object from dataset.json
    with open(filename,'r') as dataset_json:
        dataset_obj = json.load(dataset_json)

    vocabulary = dataset_obj['vocabulary']
    segments = dataset_obj['segments']
    
    # suffle the segments order
    random.shuffle(segments)

    split = int(len(segments)*ratio/100)
    test_features = { 'commands': [segment['commands'] for segment in segments[0:split]] }
    test_labels = [float(segment['label']) for segment in segments[0:split]]
    train_features = { 'commands': [segment['commands'] for segment in segments[split:]] }
    train_labels = [float(segment['label']) for segment in segments[split:]]
    
    return train_features, train_labels, test_features, test_labels, vocabulary



def _input_fn(features, labels, batch_size=1, shuffle=True, num_epochs=None):
    """
    Args:
      features: dictionary of features
      labels: list of of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in features.items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features,labels)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=len(labels))
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def main():

    # tf.enable_eager_execution()
    # print("TensorFlow version: {}".format(tf.VERSION))
    # print("Eager execution: {}".format(tf.executing_eagerly()))

    steps = 1000
    embedding_dim = 2
    learning_rate = 0.1

    data_json_name = os.path.abspath(os.path.join('MFDCA-DATA','dataset.json'))
    train_features, train_labels, test_features, test_labels, vocabulary = get_data_from_json(data_json_name)
    
    # define the commands feature column as a categorical_column_with_vocabulary_list
    commands_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="commands", vocabulary_list=vocabulary)
    
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    feature_columns = [ commands_feature_column ]

    # liner classifier
    # classifier = tf.estimator.LinearClassifier(
    #     feature_columns=feature_columns,
    #     optimizer=my_optimizer,
    # )


   
    classifier = tf.estimator.DNNClassifier(
        feature_columns=[tf.feature_column.indicator_column(commands_feature_column)],
        hidden_units=[20,20],
        optimizer=my_optimizer,
    )

    # use indicator column for commands feature
    # commands_indicator_column = tf.feature_column.indicator_column(commands_feature_column)
    # feature_columns = [ commands_indicator_column ]

    # use embedding column for commands feature
    commands_embedding_column = tf.feature_column.embedding_column(commands_feature_column, dimension=embedding_dim)
    feature_columns = [ commands_embedding_column ]

    # DNN classifier
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[20,20],
        optimizer=my_optimizer
    )
    

    try:
        classifier.train(
        input_fn=lambda: _input_fn(train_features, train_labels),
        steps=steps)

        evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn(train_features, train_labels),
        steps=steps)
        print("Training set metrics:")
        for m in evaluation_metrics:
            print (m, evaluation_metrics[m])
        print ("---")

        evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn(test_features, test_labels),
        steps=steps)

        print ("Test set metrics:")
        for m in evaluation_metrics:
            print (m, evaluation_metrics[m])
        print ("---")

    except ValueError as err:
        print(err)



if __name__ == '__main__':
    main()