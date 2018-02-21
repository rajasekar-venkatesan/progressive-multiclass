#Imports
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import accuracy_score, classification_report


#Classes
class PLT:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = None
        self.i2h = np.random.uniform(-1, 1, (input_dim, hidden_dim))
        self.h2o = None
        self.labels_set = set()
        self.labels2index_map = {}
        self.internal_vars = {}
        self.activation = self._sigmoid
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger
        pass

    def initial_batch(self, X0, y0_actual):
        self.logger.debug(f'Processing Initial Batch')
        try:
            self._update_labels(set(y0_actual))
            y0 = self._label_to_bipolar(y0_actual)
            self._calculate_weights(X0, y0)
        except:
            self.logger.error('---> !!! Error in processing initial batch !!!')
            raise ValueError
        self.logger.debug(f'Initial batch processed')

    def train(self, X, y_actual):
        batch_labels = set(y_actual.tolist())
        new_labels = batch_labels - self.labels_set
        if len(new_labels):
            try:
                self.logger.debug('Updating model architecture...')
                self._update_model_architecture(new_labels)
                self.logger.debug('Updating model architecture done.')
            except:
                self.logger.error('---> !!! Error in updating model architecture !!!')
                raise ValueError

        y = self._label_to_bipolar(y_actual)
        self._update_weights(X, y)

    def predict(self, X):
        self.logger.debug('Predicting outputs...')
        try:
            out_hidden = self.activation(np.dot(X, self.i2h))
            y_pred = np.dot(out_hidden, self.h2o)
            y_pred = np.array([np.argmax(y_pred[i, :]) for i in range(len(y_pred))])
        except:
            self.logger.error('---> !!! Error in making prediction !!!')
        self.logger.debug('Predicting outputs done.')
        return y_pred

    def get_label_indices(self, y):
        self.logger.debug('Getting label indices')
        labels = np.array([self.labels2index_map[str(label)] for label in y])
        self.logger.debug('Returning label indices')
        return labels

    def test(self, X, y):
        self.logger.debug('Testing...')
        y_pred = self.predict(X)
        y_actual = self.get_label_indices(y)
        self.logger.debug('Testing done. Generating results')
        try:
            self.logger.debug('Generating classification report...')
            report = classification_report(y_actual, y_pred)
            self.logger.debug('Generating classification report done.')
        except:
            self.logger.error('---> !!! Error in generating classification report !!!')
            raise ValueError
        try:
            self.logger.debug('Calculating accuracy score...')
            accuracy = accuracy_score(y_actual, y_pred)
            self.logger.debug('Calculating accuracy score done.')
        except:
            self.logger.error('---> !!! Error in calculating accuracy score !!!')
            raise ValueError
        return report, accuracy

    def _update_weights(self, X, y):
        try:
            M = self.internal_vars['M']
            beta = self.h2o
            H = self.activation(np.dot(X, self.i2h))
            Dr = np.eye(H.shape[0]) + np.dot(np.dot(H, M), np.transpose(H))
            Nr1 = np.dot(M, np.transpose(H))
            Nr2 = inv(Dr)
            Nr = np.dot(np.dot(np.dot(Nr1, Nr2), H), M)
            M = M - Nr
            Nr3 = y - np.dot(H, beta)
            Nr4 = np.dot(np.dot(M, np.transpose(H)), Nr3)
            beta = beta + Nr4
            self.h2o = beta
            self.internal_vars['H'] = H
            self.internal_vars['M'] = M
        except:
            self.logger.error('---> !!! Error in updating weights !!!')
            raise ValueError
        pass

    def _update_model_architecture(self, new_labels):
        self.logger.debug('Updating model architecture...')
        try:
            self._update_labels(new_labels)
            H = self.internal_vars['H']
            M = self.internal_vars['M']
            c = len(new_labels)
            N_prime = self.hidden_dim
            m = len(self.labels_set) - len(new_labels)
            b = H.shape[0]  # Previous batch size
            beta = self.h2o

            beta_tilde = np.dot(beta, np.eye(m, m + c))
            Nr1 = np.dot(M, np.transpose(H))
            Nr2 = -1 * np.ones((b, c))
            delta_beta_c = np.dot(Nr1, Nr2)
            delta_beta_mc = np.concatenate((np.zeros((N_prime, m)), delta_beta_c), axis=1)
            beta_mc = beta_tilde + delta_beta_mc
            beta = beta_mc
            self.h2o = beta
        except:
            self.logger.error('---> !!! Error in updating model architecture !!!')
            raise ValueError
        self.logger.debug('Updating model architecture done.')

    def _calculate_weights(self, X0, y0):
        try:
            H0 = self.activation(np.dot(X0, self.i2h))
            M0 = inv(np.dot(np.transpose(H0), H0))
            beta0 = np.dot(np.dot(M0, np.transpose(H0)), y0)
            H = H0
            M = M0
            self.h2o = beta0
            self.internal_vars = {'H': H, 'M': M}
        except:
            self.logger.error('---> !!! Error in calculating weights !!!')
            raise ValueError
        pass

    def _set_output_dim(self, output_dim):
        self.output_dim = output_dim
        pass

    def _update_labels(self, new_labels):
        self.logger.debug(f'Updating labels...')
        try:
            self.labels_set = self.labels_set | set(new_labels)
            for label in new_labels:
                self.labels2index_map[label] = len(self.labels2index_map)
            self._set_output_dim(len(self.labels2index_map))
        except:
            self.logger.error('---> !!! Error in updating labels !!!')
            raise ValueError
        self.logger.debug('Updating labels done.')
        pass

    def _label_to_bipolar(self, y):
        try:
            y = np.array([self.labels2index_map[label] for label in y])
        except:
            self.logger.error('---> !!! Error in converting label to bipolar !!!')
            raise ValueError

        return self._to_bipolar(y)

    def _to_bipolar(self, y):
        try:
            y_bipolar = np.ones((len(y), len(self.labels2index_map))) * -1
            for i, label in enumerate(y):
                y_bipolar[i, label] = 1
        except:
            self.logger.error('---> !!! Error in to_bipolar !!!')
            raise ValueError
        return y_bipolar

    def _sigmoid(self, data):
        try:
            result = 1 / (1 + np.exp(-1 * data))
        except:
            self.logger.error('---> !!! Error in calculating sigmoid !!!')
            raise ValueError
        return result


# Main
if __name__ == '__main__':
    pass
