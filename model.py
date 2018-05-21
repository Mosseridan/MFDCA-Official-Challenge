from __future__ import absolute_import, division, print_function

import os
import csv
import json
import random
import numpy as np
# import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

from pprint import pprint
from process_data import get_segments



def get_data_from_obj(dataset_obj, ratio=30):
    """
    Args:
      dataset_obj: an object containing the data
      ratio: the retio between the train and tests set devision 
            (10 means the test set will get 10% of the data and the train set wil get 90%)
    Returns:
      List of (train_features, train_labels, test_features, test_labels, vocabulary)
    """
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

    return get_data_from_obj(dataset_obj, ratio)



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



def train_and_test_classifier(classifier, steps, train_features, train_labels, test_features, test_labels):
    metrics = {}
    try:
        classifier.train(
            input_fn=lambda: _input_fn(train_features, train_labels),
            steps=steps)

        metrics['train_metrics'] = classifier.evaluate(
        input_fn=lambda: _input_fn(train_features, train_labels),
        steps=steps)
        print("Training set metrics:")
        for key, val in metrics['train_metrics'].items():
            print (key, val)
        print ("---")

        metrics['test_metrics'] = classifier.evaluate(
            input_fn=lambda: _input_fn(test_features, test_labels),
            steps=steps)

        print ("Test set metrics:")
        for key, val in metrics['test_metrics'].items():
            print (key, val)
        print ("---")

    except ValueError as err:
        print(err)

    return metrics



def main():

    # tf.enable_eager_execution()
    # print("TensorFlow version: {}".format(tf.VERSION))
    # print("Eager execution: {}".format(tf.executing_eagerly()))
    reps = range(2)
    steps_to_try = [1000, 2000]
    embedding_dims_to_try = [2, 10, 100]
    learning_rates_to_try = [0.1 , 0.01]
    hidden_units_to_try = [[4,4], [10,10], [20,20]]
    metrics = {}

    data_json_name = os.path.abspath(os.path.join('MFDCA-DATA','dataset.json'))
    metrics_csv_name = os.path.abspath(os.path.join('MFDCA-DATA','metrics.csv'))
    
    with open(data_json_name,'r') as dataset_json:
        dataset_obj = json.load(dataset_json)


    metrics_csv = open(metrics_csv_name, 'w', newline='')
    fieldnames = [
        'rep',
        'learning_rate',
        'steps',
        'classifier',
        'feature_column',
        'train_auc_precision_recall',
        'train_recall',
        'train_average_loss',
        'train_accuracy_baseline',
        'train_accuracy',
        'train_auc',
        'train_loss',
        'train_label/mean',
        'train_precision',
        'train_global_step',
        'train_prediction/mean',
        'test_auc_precision_recall',
        'test_recall',
        'test_average_loss',
        'test_accuracy_baseline',
        'test_accuracy',
        'test_auc',
        'test_loss',
        'test_label/mean',
        'test_precision',
        'test_global_step',
        'test_prediction/mean'
    ]
    csv_writer = csv.DictWriter(metrics_csv, fieldnames=fieldnames)
    csv_writer.writeheader()
    
    for rep in reps:
        print('\n@@ Rep: ',rep)
        metrics[rep] = {}
        train_features, train_labels, test_features, test_labels, vocabulary = get_data_from_obj(dataset_obj)
    
        # define the commands feature column as a categorical_column_with_vocabulary_list
        commands_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="commands", vocabulary_list=vocabulary)
    
        for learning_rate in learning_rates_to_try:
            print('\n@@ Learning rate: ',learning_rate)
            metrics[rep][learning_rate] = {}
            for steps in steps_to_try:
                print('\n@@ Steps: ',steps)
                metrics[rep][learning_rate][steps] = {}

                # use AdagradOptimizer with gradient clipings
                my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
                my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

                
                # linear classifier
                feature_columns = [ commands_feature_column ]
                
                classifier = tf.estimator.LinearClassifier(
                    feature_columns=feature_columns,
                    optimizer=my_optimizer,
                )

                print('\n@@ Running linear classifier: ')                
                m = train_and_test_classifier(classifier, steps, train_features, train_labels, test_features, test_labels)

                metrics[rep][learning_rate][steps]['linear_classifier'] = m
                csv_writer.writerow({
                    'rep': rep,
                    'learning_rate': learning_rate,
                    'steps': steps,
                    'classifier': 'linear classifier',
                    'feature_column': 'categorical column with vocabulary list',
                    'train_auc_precision_recall':m['train_metrics']['auc_precision_recall'],
                    'train_recall':m['train_metrics']['recall'],
                    'train_average_loss':m['train_metrics']['average_loss'],
                    'train_accuracy_baseline':m['train_metrics']['accuracy_baseline'],
                    'train_accuracy':m['train_metrics']['accuracy'],
                    'train_auc':m['train_metrics']['auc'],
                    'train_loss':m['train_metrics']['loss'],
                    'train_label/mean':m['train_metrics']['label/mean'],
                    'train_precision':m['train_metrics']['precision'],
                    'train_global_step':m['train_metrics']['global_step'],
                    'train_prediction/mean':m['train_metrics']['prediction/mean'],
                    'test_auc_precision_recall':m['test_metrics']['auc_precision_recall'],
                    'test_recall':m['test_metrics']['recall'],
                    'test_average_loss':m['test_metrics']['average_loss'],
                    'test_accuracy_baseline':m['test_metrics']['accuracy_baseline'],
                    'test_accuracy':m['test_metrics']['accuracy'],
                    'test_auc':m['test_metrics']['auc'],
                    'test_loss':m['test_metrics']['loss'],
                    'test_label/mean':m['test_metrics']['label/mean'],
                    'test_precision':m['test_metrics']['precision'],
                    'test_global_step':m['test_metrics']['global_step'],
                    'test_prediction/mean':m['test_metrics']['prediction/mean']
                })

                metrics[rep][learning_rate][steps]['dnn_classifier'] = {}
                for hidden_units in hidden_units_to_try:
                    print('\n@@ Hidden units: ',hidden_units)                                
                    metrics[rep][learning_rate][steps]['dnn_classifier'][str(hidden_units)] = {}
        
                    # DNN classifier with indicator column
                    commands_indicator_column = tf.feature_column.indicator_column(commands_feature_column)
                    feature_columns = [ commands_indicator_column ]

                    classifier = tf.estimator.DNNClassifier(
                        feature_columns=feature_columns,
                        hidden_units=hidden_units,
                        optimizer=my_optimizer
                    )

                    print('\n@@ Running dnn classifier with indicator column: ')                                
                    m = train_and_test_classifier(classifier, steps, train_features, train_labels, test_features, test_labels)
                   
                    metrics[rep][learning_rate][steps]['dnn_classifier'][str(hidden_units)]['indicator_column'] = m
                    csv_writer.writerow({
                        'rep': rep,
                        'learning_rate': learning_rate,
                        'steps': steps,
                        'classifier': 'dnn classifier with '+str(hidden_units)+' hidden_units',
                        'feature_column': 'indicator column',
                        'train_auc_precision_recall':m['train_metrics']['auc_precision_recall'],
                        'train_recall':m['train_metrics']['recall'],
                        'train_average_loss':m['train_metrics']['average_loss'],
                        'train_accuracy_baseline':m['train_metrics']['accuracy_baseline'],
                        'train_accuracy':m['train_metrics']['accuracy'],
                        'train_auc':m['train_metrics']['auc'],
                        'train_loss':m['train_metrics']['loss'],
                        'train_label/mean':m['train_metrics']['label/mean'],
                        'train_precision':m['train_metrics']['precision'],
                        'train_global_step':m['train_metrics']['global_step'],
                        'train_prediction/mean':m['train_metrics']['prediction/mean'],
                        'test_auc_precision_recall':m['test_metrics']['auc_precision_recall'],
                        'test_recall':m['test_metrics']['recall'],
                        'test_average_loss':m['test_metrics']['average_loss'],
                        'test_accuracy_baseline':m['test_metrics']['accuracy_baseline'],
                        'test_accuracy':m['test_metrics']['accuracy'],
                        'test_auc':m['test_metrics']['auc'],
                        'test_loss':m['test_metrics']['loss'],
                        'test_label/mean':m['test_metrics']['label/mean'],
                        'test_precision':m['test_metrics']['precision'],
                        'test_global_step':m['test_metrics']['global_step'],
                        'test_prediction/mean':m['test_metrics']['prediction/mean']
                    })


                    # DNN classifier with embedding column
                    metrics[rep][learning_rate][steps]['dnn_classifier'][str(hidden_units)]['embedding_column'] = {}
                    for embedding_dim in embedding_dims_to_try:
                        print('\n@@ Embedding dim: ',embedding_dim)                                                    
                        commands_embedding_column = tf.feature_column.embedding_column(commands_feature_column, dimension=embedding_dim)
                        feature_columns = [ commands_embedding_column ]
                        
                        classifier = tf.estimator.DNNClassifier(
                            feature_columns=feature_columns,
                            hidden_units=hidden_units,
                            optimizer=my_optimizer
                        )

                        print('\n@@ Running dnn classifier with embedding column: ')                                                    
                        m = train_and_test_classifier(
                            classifier,
                            steps,
                            train_features,
                            train_labels,
                            test_features,
                            test_labels)
                        
                        metrics[rep][learning_rate][steps]['dnn_classifier'][str(hidden_units)]['embedding_column'][str(embedding_dim)] = m
                        csv_writer.writerow({
                            'rep': rep,
                            'learning_rate': learning_rate,
                            'steps': steps,
                            'classifier': 'dnn classifier with '+str(hidden_units)+' hidden_units',
                            'feature_column': str(embedding_dim)+' dim embedding column',
                            'train_auc_precision_recall':m['train_metrics']['auc_precision_recall'],
                            'train_recall':m['train_metrics']['recall'],
                            'train_average_loss':m['train_metrics']['average_loss'],
                            'train_accuracy_baseline':m['train_metrics']['accuracy_baseline'],
                            'train_accuracy':m['train_metrics']['accuracy'],
                            'train_auc':m['train_metrics']['auc'],
                            'train_loss':m['train_metrics']['loss'],
                            'train_label/mean':m['train_metrics']['label/mean'],
                            'train_precision':m['train_metrics']['precision'],
                            'train_global_step':m['train_metrics']['global_step'],
                            'train_prediction/mean':m['train_metrics']['prediction/mean'],
                            'test_auc_precision_recall':m['test_metrics']['auc_precision_recall'],
                            'test_recall':m['test_metrics']['recall'],
                            'test_average_loss':m['test_metrics']['average_loss'],
                            'test_accuracy_baseline':m['test_metrics']['accuracy_baseline'],
                            'test_accuracy':m['test_metrics']['accuracy'],
                            'test_auc':m['test_metrics']['auc'],
                            'test_loss':m['test_metrics']['loss'],
                            'test_label/mean':m['test_metrics']['label/mean'],
                            'test_precision':m['test_metrics']['precision'],
                            'test_global_step':m['test_metrics']['global_step'],
                            'test_prediction/mean':m['test_metrics']['prediction/mean']
                        })

    
    metrics_csv.close()
    
    print('\n@@ Writing metrics to metrics.json')                                                        
    with open(os.path.abspath(os.path.join('MFDCA-DATA','metrics.json')), 'w') as outfile:
        json.dump(metrics, outfile)


    # try:
    #     classifier.train(
    #     input_fn=lambda: _input_fn(train_features, train_labels),
    #     steps=steps)

    #     evaluation_metrics = classifier.evaluate(
    #     input_fn=lambda: _input_fn(train_features, train_labels),
    #     steps=steps)
    #     print("Training set metrics:")
    #     for m in evaluation_metrics:
    #         print (m, evaluation_metrics[rep][m])
    #     print ("---")

    #     evaluation_metrics = classifier.evaluate(
    #     input_fn=lambda: _input_fn(test_features, test_labels),
    #     steps=steps)

    #     print ("Test set metrics:")
    #     for m in evaluation_metrics:
    #         print (m, evaluation_metrics[rep][m])
    #     print ("---")

    # except ValueError as err:
    #     print(err)





    # pprint(classifier.get_variable_names())
    # print(classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/commands_embedding/embedding_weights').shape)
    
    # embedding_matrix = classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

    # for cmd_index in range(len(vocabulary)):
    #     # Create a one-hot encoding for our term. It has 0s everywhere, except for
    #     # a single 1 in the coordinate that corresponds to that term.
    #     cmd_vector = np.zeros(len(vocabulary))
    #     cmd_vector[cmd_index] = 1
    #     # We'll now project that one-hot vector into the embedding space.
    #     embedding_xy = np.matmul(cmd_vector, embedding_matrix)
    #     plt.text(embedding_xy[0],
    #             embedding_xy[1],
    #             vocabulary[cmd_index])

    #     # Do a little setup to make sure the plot displays nicely.
    #     plt.rcParams["figure.figsize"] = (15, 15)
    #     plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    #     plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    #     plt.show() 



if __name__ == '__main__':
    main()