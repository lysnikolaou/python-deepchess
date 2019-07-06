import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import enum
from itertools import product

import joblib
import chess
import bitstring
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from dbn.tensorflow import UnsupervisedDBN

from utils import bitify


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DBN_NETWORK_FILENAME = 'dbn_model.pkl'
MLP_NETWORK_FILENAME = 'mlp_model.pkl'
DATA_FILENAME = 'data_ready.csv'


def train_dbn(X):
    dbn_network = UnsupervisedDBN(
        hidden_layers_structure=[773, 600, 400, 200, 100],
        activation_function='relu',
        learning_rate_rbm=5*1e-3,
        n_epochs_rbm=200
    )
    dbn_network.fit(X)
    dbn_network.save(DBN_NETWORK_FILENAME)


def train_mlp(X, Y):
    dbn_network = UnsupervisedDBN.load(DBN_NETWORK_FILENAME)
    first_positions = dbn_network.transform(X[:, 0])
    second_positions = dbn_network.transform(X[:, 1])
    X = np.concatenate((first_positions, second_positions), axis=1)
    print(f'Shape after Concatenation = {X.shape}')

    mlp_network = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50, 2),
        learning_rate='constant',
        learning_rate_init=1e-2,
        max_iter=1000
    )
    mlp_network.fit(X, Y)
    joblib.dump(mlp_network, MLP_NETWORK_FILENAME)


def predict_mlp(first, second):
    First = np.array(first)
    Second = np.array(second)
    dbn_network: UnsupervisedDBN = UnsupervisedDBN.load(DBN_NETWORK_FILENAME)
    mlp_network: MLPClassifier = joblib.load(MLP_NETWORK_FILENAME)
    first_position = dbn_network.transform(First)
    second_position = dbn_network.transform(Second)
    X = np.concatenate((first_position, second_position))
    result = mlp_network.predict(X.reshape(1, -1))
    return result.flatten().tolist()


def extract_mlp_data(data):
    returned_data = list()
    returned_labels = list()
    for data_point in data:
        first_position = bitify(data_point[0].split(';')[0])
        second_position = bitify(data_point[1].split(';')[0])

        returned_data.append((first_position, second_position))
        returned_labels.append((1, 0))

        returned_data.append((second_position, first_position))
        returned_labels.append((0, 1))

    return returned_data, returned_labels


def extract_dbn_data(white_wins, black_wins):
    return np.concatenate((
        [bitify(i.split(';')[0]) for i in white_wins],
        [bitify(i.split(';')[0]) for i in black_wins]
    ))


def gather_data():
    with open(DATA_FILENAME, 'r') as file:
        raw_data = list(map(str.strip, file.readlines()))

    white_wins = list(filter(lambda e: '1-0' in e, raw_data))[:500]
    black_wins = list(filter(lambda e: '0-1' in e, raw_data))[:500]
    
    data = product(white_wins, black_wins)
    mlp_data, mlp_labels = extract_mlp_data(data)

    dbn_data = extract_dbn_data(white_wins, black_wins)
    return dbn_data, np.array(mlp_data), np.array(mlp_labels)


def test_mlp(X, Y):
    dbn_network: UnsupervisedDBN = UnsupervisedDBN.load(DBN_NETWORK_FILENAME)
    mlp_network: MLPClassifier = joblib.load(MLP_NETWORK_FILENAME)
    first_positions = dbn_network.transform(X[:, 0])
    second_positions = dbn_network.transform(X[:, 1])
    X = np.concatenate((first_positions, second_positions), axis=1)
    return mlp_network.score(X, Y)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'predict':
        first = list(map(int, sys.argv[2][1:-1].split(',')))
        second = list(map(int, sys.argv[3][1:-1].split(',')))
        result = predict_mlp(first, second)
        print(str(result))
        sys.exit(0)

    print("Gathering data in memory...")
    dbn_data, mlp_data, mlp_labels = gather_data()
    print("Gathering Data Done!")
    
    print("Splitting data in Training and Test Data...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        mlp_data,
        mlp_labels,
        test_size=0.25
    )
    print("Splitting Done!")

    print("Training Deep Belief Network...")
    train_dbn(dbn_data)
    print("Training DBN Done!")

    print("Training Multi-Layer Perceptron...")
    train_mlp(X_train, Y_train)
    print("Training MLP Done!")

    print("Testing MLP...")
    print(f"Score = {test_mlp(X_test, Y_test)}")


if __name__ == '__main__':
    main()
