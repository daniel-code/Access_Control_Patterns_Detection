"""
.. module:: path
    :synopsis: Path predict for Access_control_patterns_detection access control
        This module will predict next gate by previous record
        If gate is predicted wrong, employee may change his route in company

.. moduleauthor:: daniel-code
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder

from .utils import check_path, convert_IOcode


class PathModel:
    def __init__(self, n_estimators: int = 20, n_jobs: int = 4, random_state: int = 0):
        """
        :param n_estimators: The number of trees in the forest.
        :param n_jobs (default=4):integer, optional
        The number of jobs to run in parallel for both fit and predict.
        If -1, then the number of jobs is set to the number of cores.

        :param random_state: int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state)

    def data_encoding(self, raw_data: pd.DataFrame, building_num: int,
                      gates_code_table: dict) -> np.array:
        """
        Encode raw record data from database

        :param gates_code_table: gate code table for mapping code
        :param building_num: total buildings
        :param raw_data: raw data from DataTable.get_raw_record_data()

        :return:
            - data_list : Feature of encode data
            - target_list : gate label of encode data
        """
        week_data = raw_data['datetime'].dt.weekday.rename('week')
        raw_data = raw_data.join(week_data)
        raw_data = raw_data.reset_index().drop(columns=['index'])
        data_list = pd.DataFrame()
        #############################
        # Feature Encoding          #
        #############################
        # gate one hot encoding
        gate_one_hot_list = np.arange(len(gates_code_table)).reshape(-1, 1)
        gate_encoder = OneHotEncoder()
        gate_encoder.fit(gate_one_hot_list)

        week_one_hot_list = np.arange(7).reshape(-1, 1)
        week_encoder = OneHotEncoder()
        week_encoder.fit(week_one_hot_list)

        building_one_hot_list = np.arange(1, building_num + 1).reshape(-1, 1)
        building_encoder = OneHotEncoder()
        building_encoder.fit(building_one_hot_list)

        gatecode = raw_data['building'].str.cat([raw_data['floor'], raw_data['IO']], sep='-').apply(
            lambda x: gates_code_table[x] if x in gates_code_table else 0).rename('gate').astype(int)
        raw_data['gate'] = gatecode
        raw_data['next_gate'] = gatecode.shift(-1)

        gatecode = raw_data['gate']
        gatecode_onehotcode = gate_encoder.transform(gatecode.values.reshape(-1, 1)).toarray()
        gatecode_onehotcode = pd.DataFrame(gatecode_onehotcode, dtype='int').add_prefix('gate_')

        # weekday one hot encoding
        weekdaycode = week_encoder.transform(raw_data['week'].values.reshape(-1, 1)).toarray()
        weekdaycode = pd.DataFrame(weekdaycode, dtype='int').add_prefix('weekday_')

        # building one hot encoding
        buildingcode = raw_data['building'].astype(int)
        buildingcode_onehotcode = building_encoder.transform(buildingcode.values.reshape(-1, 1)).toarray()
        buildingcode_onehotcode = pd.DataFrame(buildingcode_onehotcode, dtype='int').add_prefix('building_')

        # Time feature
        data_list['hour'] = raw_data['datetime'].apply(lambda x: x.hour / 24)
        data_list['minute'] = raw_data['datetime'].apply(lambda x: x.minute / 60)
        data_list['second'] = raw_data['datetime'].apply(lambda x: x.second / 60)

        # IO code
        IOcode = raw_data['IO'].apply(lambda x: convert_IOcode(x))
        # join feature
        data_list = data_list.join(other=[IOcode, weekdaycode, buildingcode_onehotcode, gatecode_onehotcode])
        # match order
        data_list = data_list.dropna(how='any')
        target_list = raw_data['next_gate']

        data_list = data_list.values
        target_list = target_list.values.flatten()
        return data_list, target_list

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Build model for train set (X,y)

        Reference: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        :param X: array-like or sparse matrix of shape = [n_samples, n_features]

        :param y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values class labels in classification

        :return: None
        """
        self.model.fit(X=X, y=y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in the forest, weighted by their probability estimates.
        That is, the predicted class is the one with highest mean probability estimate across the trees.

        Reference: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        :param X: array-like or sparse matrix of shape = [n_samples, n_features]

        :return: The predicted classes.
        """
        return self.model.predict(X=X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample
        that each label set be correctly predicted.

        Reference: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        :param X: array-like, shape = (n_samples, n_features)
            Test samples.

        :param y: array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        :return: score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        return self.model.score(X=X, y=y)

    def save_model(self, filename: str):
        """
        Save model into file

        Reference: https://pythonhosted.org/joblib/generated/joblib.dump.html#joblib.dump

        :param filename: str
            The path of the file in which it is to be stored.

        :return: None

        """
        # filename contain folder
        check_path(filename)
        joblib.dump(value=self.model, filename=filename)

    def load_model(self, filename: str):
        """
        Load from file

        :param filename: str
            The path of the file in which it is to be loaded

        :return: None
        """
        if os.path.exists(filename):
            self.model = joblib.load(filename=filename)
        else:
            raise Exception('{} is not existed!'.format(filename))

    def to_csv(self, filename: str, y_raw_data, y_true, y_pred):
        """
        Output y_raw_data abnormal predict data into abnormal database

        :param filename: ouptut csv filename
        :param y_raw_data: y raw data
        :param y_true: true gate code
        :param y_pred: predict gate code

        :return: None

        """
        output_list = []
        for index in range(len(y_raw_data)):
            if y_true[index] != y_pred[index]:
                output_list.append(
                    [index, y_raw_data.iloc[index]['datetime'], y_raw_data.iloc[index]['employee_ID'], y_true[index],
                     y_pred[index]])
        if len(output_list) > 0:
            output_list = pd.DataFrame(output_list, columns=['id',
                                                             'datetime',
                                                             'employee_ID',
                                                             'real_gate_code',
                                                             'predict_gate_code'])
            if filename is not None:
                output_list.to_csv(filename)

    def plot_output(self, y_true: np.ndarray, y_pred: np.ndarray, filename: str = None):
        """
        Plot to_csv png

        :param y_true: real gate code
        :param y_pred: model predict gate code
        :param filename: if not None value, save figure as filename
        :return: None
        """
        error_item = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                error_item.append([i, y_true[i]])

        plt.subplots(figsize=(20, 10))
        plt.title("Gates Prediction RandomForestClassifier")
        plt.xlabel("Record Order")
        plt.ylabel("Gate")
        plt.plot(y_pred, 'C0o', label='predict')
        if len(error_item) > 0:
            plt.plot(np.array(error_item)[:, 0], np.array(error_item)[:, 1], 'rx', markersize=6, label='error')
        plt.legend(loc='best')
        plt.grid()
        if filename is not None:
            plt.savefig(filename)
        plt.show()
