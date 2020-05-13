"""
.. module:: staytime
   :synopsis: Find abnormal stay-time raw_data used Kmeans-clustering
       There are 3 encode-type for find abnormal raw_data
        encode_type = 0 : Only consider stay outside time
        encode_type = 1 : Consider stay inside and outside time
        encode_type = 2 [Default]: Consider stay inside, outside time and record time

.. moduleauthor:: daniel-code
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

from .utils import check_path


class StayTimeModel:

    def __init__(self, encode_type=2):
        """

        :param encode_type:
            - 0 :  Only consider stay outside time [staytime, staytime]
            - 1 :  Consider stay inside and outside time [pre_staytime, staytime]
            - 2 [Default] :  Consider stay inside, outside time and record time [pre_staytime, staytime, time]
        """
        assert 0 <= encode_type <= 2, 'Encode Type Error, encode_type must between 0~2.'
        self.encode_type = encode_type
        self.model = pd.DataFrame()
        self.normal_model = pd.DataFrame()

    # fit 1-D data_list into 2-D data_list for kmean clustering
    def data_encoding(self, raw_data, time_limit=0, IO_limit=True):
        """
        Encode raw data from DataTable.get_raw_record_data() into different encode type

        :param raw_data: raw data from DataTable.get_raw_record_data()
        :param time_limit: 0:no limit time,other positive integer will limite staytime
        :param IO_limit:
            - True : only use outside staytime.
            - False(Default) : use both outside and inside staytime

        :return: 2-D encode raw_data array
        """
        assert len(raw_data) > 0, 'raw data is empty'
        encode_df = raw_data.drop_duplicates(subset=['datetime']).copy()

        staytime = encode_df['datetime'] - encode_df['datetime'].shift(1)
        encode_df['staytime'] = staytime.dt.seconds

        encode_df['pre_staytime'] = encode_df['staytime'].shift(1)
        # calculate seconds of day of record
        encode_df['time'] = (encode_df['datetime'] - encode_df['datetime'].dt.floor('D')).dt.total_seconds()

        encode_data = pd.DataFrame()
        df = encode_df.copy()
        # convert IO code
        if IO_limit:
            # only select stay outside record
            df = df[df['IO'].str.contains('I', na=False) == True]
        # encode in different type
        if self.encode_type == 0:
            encode_data['pre_staytime'] = df['staytime']
            encode_data['staytime'] = df['staytime']
        elif self.encode_type == 1:
            encode_data['pre_staytime'] = df['pre_staytime']
            encode_data['staytime'] = df['staytime']
        elif self.encode_type == 2:
            encode_data['pre_staytime'] = df['pre_staytime']
            encode_data['staytime'] = df['staytime']
            encode_data['time'] = df['time']
        # select time limit (0:not limit)
        if time_limit > 0:
            encode_data = encode_data[encode_data.staytime < time_limit]
        encode_data = encode_data.dropna(how='any').values
        return encode_data, encode_df

    def _find_init_K(self, X, max_num_clusters=10, verbose=False):
        """
        find initial min k form 2 to max_num_clusters

        :param X: input raw_data 2-D array encode raw_data form data_encoding()
        :param max_num_clusters: max initial clusters number
        :param verbose: control visible of detail
            - True : show detail of processing
            - False : show nothing

        :return:
        init_k : min initial K
        key_value : AD-test grade for K=2~max_num_clusters

        """
        key_value = []
        item_in_clusters = {}
        # check length of X
        if 2 < len(X) < max_num_clusters:
            max_num_clusters = len(X)
        elif len(X) < 2:
            raise Exception('There have not enough data for clustering !!')
        # calculate AD-test form 2 to max_num_clusters
        for k in range(2, max_num_clusters + 1):
            item_in_clusters.clear()
            labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
            n = 0
            for item in labels:
                if item in item_in_clusters:
                    item_in_clusters[item].append(X[n])
                else:
                    item_in_clusters[item] = [X[n]]
                n += 1
            sum = 1.0
            for item in item_in_clusters:
                points = np.array(item_in_clusters[item])[:, 0]
                points2 = np.array(item_in_clusters[item])[:, 1]
                points.sort()
                points2.sort()
                if len(points) > 1:
                    result = scipy.stats.anderson(points)
                    result2 = scipy.stats.anderson(points2)
                    sum += math.sqrt(result[0] ** 2 + result2[0] ** 2)
            key_value.append(math.log10(sum))
        # shift index
        if len(key_value) > 0:
            init_k = key_value.index(min(key_value)) + 2
        else:
            init_k = 1
        if verbose:
            print("init_k => " + str(init_k))
            print(key_value)
        return init_k, key_value

    # Density-based K-means clustering
    def _Kmeans_clustering(self, X, init_k=2, rate=1.0, e=3.69):
        """
        kmean clustering use spilte and merge base on AD-test value

        :param X: input raw_data 2-D stay time array
        :param init_k: initial K
        :param rate: for widen AD-test threshold in next recursive step to avoid too small cluster
        :param e: AD-test threshold

        :return: clusters for X raw_data

        """
        item_in_clusters = {}
        output_clusters = {}
        labels = KMeans(n_clusters=init_k, random_state=0).fit_predict(X)
        n = 0
        for item in labels:
            if item in item_in_clusters:
                item_in_clusters[item].append(X[n])
                output_clusters[item].append(X[n])
            else:
                item_in_clusters[item] = [X[n]]
                output_clusters[item] = [X[n]]
            n += 1
        for item in item_in_clusters:
            points = np.array(item_in_clusters[item])[:, 0]
            points2 = np.array(item_in_clusters[item])[:, 1]
            points.sort()
            points2.sort()
            if len(points) > 1 and len(points2) > 1:
                result = scipy.stats.anderson(points)
                result2 = scipy.stats.anderson(points2)
                if result[0] > e or result2[0] > e:
                    newClusters = self._Kmeans_clustering(item_in_clusters[item], init_k=2, rate=rate, e=e * rate)
                    del output_clusters[item]
                    keys = []
                    for key in output_clusters.keys():
                        keys.append(key)
                    keys.sort(reverse=True)
                    index = keys[0]
                    for key in newClusters:
                        index += 1
                        output_clusters[index] = newClusters[key]
        return output_clusters

    def _check_raw_data(self, X, abnormal_list):
        """
        abnormal raw_data check for encode_type = 1,use in _find_key_raw_data

        :param X: one row of rawdata DataFrame
        :param abnormal_list: abnormal array
        :return:
            - True: x in abnormal;
            - False: x not in abnormal

        """
        for item in abnormal_list:
            if self.encode_type == 1:
                if np.all([X['pre_staytime'], X['staytime']] == item):
                    return True
            elif self.encode_type == 2:
                if np.all([X['pre_staytime'], X['staytime'],
                           X['time']] == item):
                    return True
        return False

    def _find_key_raw_data(self, raw_data: pd.DataFrame, key_list):
        """
        search key raw_data from rawdata DataFrame

        :param raw_data: pandas DataFrame rawdata
        :param key_list: key raw_data for search

        :return:
            results_pre : previous key access record
            results : key access record

        """
        if len(key_list) == 0:
            results = pd.DataFrame()
            results_pre = pd.DataFrame()
        else:
            raw_data = raw_data.reset_index(drop=True)
            results = raw_data[raw_data.apply(
                lambda x: True if x.staytime in key_list[:, 1] and x['IO'][0] == 'I' else False,
                axis=1)]
            if self.encode_type != 0:
                results = results[results.apply(lambda x: self._check_raw_data(x, key_list), axis=1)]
            pre_index = results.index.values - 1
            results_pre = raw_data.iloc[pre_index]
        return results_pre, results

    def fit(self, data, max_num_clusters=10, rate=1.0, e=3.69, std_drop_limit=5, verbose=False):
        """
        Build model of stay time

        :param data: array-like or sparse matrix of shape = [n_samples, n_features]
        :param max_num_clusters: max initial clusters number
        :param rate: for widen AD-test threshold in next recursive step to avoid too small cluster
        :param e: AD-test threshold
        :param std_drop_limit: drop max and min value in cluster, and calculate std of cluster
        :param verbose:
            - True : show detail of processing
            - False : show nothing

        :return: None
        """
        init_k, key_value = self._find_init_K(data, max_num_clusters)
        clusters = self._Kmeans_clustering(data, init_k, rate, e)
        model_list = []
        normal_model_list = []
        for item in clusters:
            distance = []
            distance.append(np.array(clusters[item])[:, 0].tolist())
            distance.append(np.array(clusters[item])[:, 1].tolist())
            center_list = [len(clusters[item]), np.mean(distance[0]), np.std(distance[0]), np.mean(distance[1]),
                           np.std(distance[1])]
            if self.encode_type == 2:
                distance.append(np.array(clusters[item])[:, 2].tolist())
            if len(clusters[item]) > std_drop_limit:
                distance[0].remove(max(distance[0]))
                distance[0].remove(min(distance[0]))
                distance[1].remove(max(distance[1]))
                distance[1].remove(min(distance[1]))
            clean_center_list = [len(clusters[item]), np.mean(distance[0]), np.std(distance[0]), np.mean(distance[1]),
                                 np.std(distance[1])]
            if self.encode_type == 2:
                center_list.extend([np.mean(distance[2]), np.std(distance[2])])
                clean_center_list.extend([np.mean(distance[2]), np.std(distance[2])])
            model_list.append(np.array(center_list).flatten())
            normal_model_list.append(np.array(clean_center_list).flatten())
        self.model = pd.DataFrame(model_list)
        self.normal_model = pd.DataFrame(normal_model_list)
        if verbose:
            print(self.model)
            print(self.normal_model)

    def predict(self, data, std_threshold=2.0, min_cluster_size=3):
        """
        calculate abnormal state of data

        :param data: array-like or sparse matrix of shape = [n_samples, n_features]
        :param std_threshold: threshold of distance between data and cluster mean
        :param min_cluster_size: min cluster size, if less then size whole group will be abnormal

        :return:
            - predict_list : predict abnormal state

                - -1 : abnormal data

                - 1 : normal data

            - labels: group id of data
        """
        assert len(data) > 0, 'X is empty'
        assert isinstance(self.model, pd.DataFrame), 'not fit'
        if self.encode_type == 2:
            cluster_centers_ = self.model[[1, 3, 5]].values
        else:
            cluster_centers_ = self.model[[1, 3]].values
        labels, mindist = pairwise_distances_argmin_min(
            X=data, Y=cluster_centers_, metric='euclidean', metric_kwargs={'squared': True})
        cluster_centers_ = self.normal_model.iloc[labels]
        predict_list = []
        for index in range(len(data)):
            cluster_centers_data = cluster_centers_.iloc[index]
            if cluster_centers_data[0] + 1 <= min_cluster_size:
                predict_list.append(-1)
            elif math.fabs(data[index][0] - cluster_centers_data[1]) > cluster_centers_data[2] * std_threshold and \
                    math.fabs(data[index][1] - cluster_centers_data[3]) > cluster_centers_data[4] * std_threshold:
                predict_list.append(-1)
            else:
                predict_list.append(1)
        predict_list = np.array(predict_list)
        return predict_list, labels

    def score(self, X):
        """
        mean distance between X and cluster mean

        :param X: array-like or sparse matrix of shape = [n_samples, n_features]
        :return: mean distance between X and cluster mean
        """
        cluster_centers_ = self.model[[1, 3]].values
        if self.encode_type == 2:
            cluster_centers_ = self.model[[1, 3, 5]].values
        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=cluster_centers_, metric='euclidean', metric_kwargs={'squared': True})
        return np.array(mindist).mean()

    def save_model(self, filename):
        """
        Save model into file

        :param filename: str
            The path of the file in which it is to be stored.

        :return: None
        """
        # filename contain folder
        check_path(filename)
        model_file = pd.HDFStore(filename)
        model_file.put('encode_type', pd.DataFrame([self.encode_type]))
        model_file.put('model', self.model)
        model_file.put('normal_model', self.normal_model)
        model_file.close()

    def load_model(self, filename):
        """
        Load from file

        :param filename: str
            The path of the file in which it is to be loaded

        :return: None
        """
        if os.path.exists(filename):
            model_file = pd.HDFStore(filename)
            self.encode_type = model_file['encode_type'].values[0][0]
            self.model = model_file['model']
            self.normal_model = model_file['normal_model']
            model_file.close()
        else:
            raise Exception('{} is not existed!'.format(filename))

    def plot_output(self, encode_data, x_limit=(0, 0), y_limit=(0, 0), abnormal_verbose=True,
                    center_verbose=True, file_name=None):
        """
        Plot output png

        :param x_limit: x-axis limit
        :param y_limit: y-axis limit

        :param encode_data:  encode data
        :param abnormal_verbose:
            - True : show abnormal data
            - False : not show abnormal data

        :param center_verbose:
            - True : show group center
            - False : not show group center

        :param file_name: file path for stored

        :return: None
        """
        assert len(encode_data) > 0, 'Empty encode data!'
        y_pred, labels = self.predict(data=encode_data)
        y_pred = pd.Series(y_pred)
        labels = pd.Series(labels)
        labels_list = labels.value_counts().index.values
        abnormal_index = y_pred[y_pred == -1].index.values
        abnormal_data = encode_data[abnormal_index]
        plt.figure(figsize=(20, 20))
        plt.title('Stay Time\nEncode Type = {}'.format(self.encode_type))
        for item in labels_list:
            index_list = labels[labels == item].index.values
            data = encode_data[index_list]
            plt.scatter(data[:, 0], data[:, 1])
        if center_verbose:
            plt.scatter(self.normal_model[1], self.normal_model[3], c='r', marker='+', s=500, alpha=0.3,
                        label='cluster center')
        if abnormal_verbose:
            plt.scatter(abnormal_data[:, 0], abnormal_data[:, 1], alpha=1, s=100, color='red', marker='x',
                        label='abnormal data')
        if file_name is not None:
            plt.savefig(file_name, transparent=True)
        plt.xlabel('Stay Inside Time (s)')
        plt.ylabel('Stay Outside Time (s)')
        if x_limit[1] * y_limit[1] != 0:
            plt.xlim(x_limit)
            plt.ylim(y_limit)
        plt.grid(True)
        plt.show()

    def to_csv(self, filename, raw_data, encode_data, abnormal_list, mode='w', verbose=False):
        """
        Output abnormal to csv file

        :param filename: output file path
        :param raw_data: raw data
        :param abnormal_list: abnormal data list
        :param mode: Pandas write mode
        :param verbose:
            - True :  show abnormal data
            - False : not show abnormal data

        :return: None

        """
        abnormal_list = pd.Series(abnormal_list)
        abnormal_index = abnormal_list[abnormal_list == -1].index.values
        if len(abnormal_index) > 0:
            abnormal_list = encode_data[abnormal_index]
            results_pre, results = self._find_key_raw_data(raw_data, abnormal_list)
            results_pre['SN'] = 0
            results['SN'] = 1
            output = pd.concat([results_pre, results]).sort_index()
            output = output.drop(['pre_staytime', 'time'], axis=1)
            output.to_csv(path_or_buf=filename, header=True, mode=mode)
            if verbose:
                print(output)
