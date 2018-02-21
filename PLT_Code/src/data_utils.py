# Imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Classes
class Data:
    def __init__(self):
        self.logger = None
        self.src_fname = None
        self.data = None
        self.feats = None
        self.num_feats = None
        self.labels = None
        self.scaler = None
        self.scaled_data = None
        self.train_data = {}
        self.test_data = {}
        self.initial_batch_size = None
        self.training_batch_size = None
        pass

    def set_logger(self, logger):
        self.logger = logger
        pass

    def load_csv(self, fname, header='infer'):
        self.src_fname = fname
        self.logger.debug(f'Loading dataset from file: {fname}...')
        try:
            data_from_file = pd.read_csv(fname, header=header)
        except:
            self.logger.error(f'---> !!! Error in reading file: {fname} !!!')
        self.data = data_from_file.values
        self.logger.debug(f'Loaded {self.data.shape[0]} samples from the file {fname.split("/")[-1]}')
        self.logger.info('Loading dataset done.')
        return self.data

    def get_feats_labels(self, data=None, label_column='last'):
        self.logger.info('Extracting features and labels...')
        if data is None:
            data = self.data
        if label_column == 'last':
            self.feats = data[:, :-1]
            self.labels = data[:, -1]
            self.num_feats = self.feats.shape[1]
        elif label_column == 'first':
            self.feats = data[:, 1:]
            self.labels = data[:, 1]
            self.num_feats = self.feats.shape[1]
        else:
            self.logger.error(f'---> !!! Label column not valid !!!')
            raise ValueError

        self.logger.debug(f'Extracted {self.feats.shape[1]} features')
        self.logger.info('Extracting features and labels done.')
        return self.feats, self.labels

    def scale_features(self, feats=None, scale_type='minmax'):
        self.logger.info('Scaling features...')
        self.logger.debug(f'scaling type: {scale_type}')
        try:
            if feats is None:
                feats = self.feats
            if scale_type == 'minmax':
                self.logger.info('Performing minmax scaler')
                scaler = MinMaxScaler()
                self.scaled_feats = scaler.fit_transform(feats)
            elif scale_type == 'std':
                self.logger.info('Performing standard scaler')
                scaler = StandardScaler()
                self.scaled_feats = scaler.fit_transform(feats)
            elif scale_type == None:
                self.logger.info('No scaling')
                self.scaled_feats = feats
            else:
                self.logger.warning('---> !!! Scaler type not valid. Scaling not performed. Returning original values !!!')
                self.scaled_feats = feats
        except:
            self.logger.error('---> !!! Error in scaling features !!!')
            raise ValueError

        self.logger.info('Scaling features done.')
        return self.scaled_data

    def split_train_test(self, feats=None, labels=None, test_ratio=0.1):
        self.logger.info(f'Splitting into train and test data... Test ratio: {test_ratio}')
        if not (test_ratio > 0 and test_ratio < 1):
            self.logger.error(f'---> !!! Test ratio {test_ratio} is not valid !!!')
            raise AssertionError

        try:
            if feats is None:
                feats = self.scaled_feats
            if labels is None:
                labels = self.labels

            test_size = int(len(feats) * test_ratio)
            self.logger.info(f'Test size: {test_size}')
            train_X = feats[:-test_size]
            train_y = labels[: -test_size]
            test_X = feats[-test_size:]
            test_y = labels[-test_size:]

            self.logger.debug(f'train_X: {train_X.shape} train_y: {train_y.shape}'
                              f'test_X: {test_X.shape} test_y: {test_y.shape}')
            self.train_data = {'X': train_X, 'y': train_y}
            self.test_data = {'X': test_X, 'y': test_y}
            self.logger.info('Splitting into train and test data complete.')

        except:
            self.logger.error('---> !!! Error in Train Test Split !!!')
            raise ValueError

        return train_X, train_y, test_X, test_y

    def set_initial_batch_size(self, initial_batch_size):
        self.logger.debug(f'Setting initial batch size to {initial_batch_size}')
        self.initial_batch_size = initial_batch_size
        pass

    def set_training_batch_size(self, training_batch_size):
        self.logger.debug(f'Setting training batch size to {training_batch_size}')
        self.training_batch_size = training_batch_size
        pass

    #Generators
    def fetch_train_data(self, initial_batch_size=None, training_batch_size=None, train_X=None, train_y=None):
        if initial_batch_size is None:
            initial_batch_size = self.initial_batch_size
        if training_batch_size is None:
            training_batch_size = self.training_batch_size
        if train_X is None:
            train_X = self.train_data['X']
        if train_y is None:
            train_y = self.train_data['y']
        self.logger.debug(f'Fetching data. Initial batch size: {initial_batch_size},'
                          f'Training batch size: {training_batch_size}, '
                          f'Train_X: {train_X.shape}, Train_y: {train_y.shape}')

        batch_id = 0
        X = train_X[:initial_batch_size]
        y = train_y[:initial_batch_size]
        self.logger.debug(f'Yielding batch {batch_id} data, X: {X.shape}, y: {y.shape}')
        yield batch_id, X, y

        train_X = train_X[initial_batch_size:]
        train_y = train_y[initial_batch_size:]

        start = 0
        end = training_batch_size
        while end <= train_X.shape[0]:
            batch_id += 1
            X = train_X[start:end, :]
            y = train_y[start:end]
            start += training_batch_size
            end += training_batch_size
            self.logger.debug(f'Yielding batch {batch_id} data, X: {X.shape}, y: {y.shape}')
            yield batch_id, X, y


# Main
if __name__ == '__main__':
    pass