#Imports
import pandas as pd
import numpy as np
from numpy.linalg import pinv, inv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#Functions
def to_bipolar(y, labels2index_map):
    y_bipolar = np.ones((len(y), len(labels2index_map))) * -1
    for i, label in enumerate(y):
        y_bipolar[i, label] = 1
    return y_bipolar

def sigmoid(data):
    result = 1 / (1 + np.exp(-1 * data))
    return result

def get_batch_data_train(batch_size, train_X, train_y_bipolar):
    start = 0
    end = batch_size
    while end <= train_X.shape[0]:
        X = train_X[start:end, :]
        y = train_y_bipolar[start:end]
        start += batch_size
        end += batch_size
        yield X, y

#Main
if __name__ == '__main__':
    print('---PROGRESSIVE ELM - MULTI-CLASS---')
    fname = 'iris_plt.csv'
    print(f'Loading data from file: {fname}')
    data = pd.read_csv(fname).values
    # np.random.shuffle(data)
    print(f'{data.shape[0]} samples loaded with {data.shape[1] - 1} features')

    feats = data[:, :-1]
    labels_raw = data[:, -1]
    # labels_set = set(labels_raw)
    # print(f'Labels are: {labels_set}')
    # labels2index_map = {label: ind for ind, label in enumerate(labels_set)}
    # print(f'Labels to index map: {labels2index_map}')
    # labels = np.array([labels2index_map[label] for label in labels_raw])

    scaler = MinMaxScaler()
    feats = scaler.fit_transform(feats)
    print('Scaling Features Done')

    # train_X, test_X, train_y, test_y = train_test_split(feats, labels_raw, test_size=0.1, random_state=42)
    test_size = 15
    train_X = feats[:-15]
    test_X = feats[-15:]
    train_y = labels_raw[:-15]
    test_y = labels_raw[-15:]

    print('Divided into training and testing set')
    print(f'Training set has {train_X.shape[0]} samples')

    # train_y_bipolar = to_bipolar(train_y, labels2index_map)
    # test_y_bipolar = to_bipolar(test_y, labels2index_map)
    # print('Converted labels to bipolar')

    nHidden = 10
    N0 = 30
    batch_size = 1

    print('Begin Training...')
    #ELM Training
    ##Initial Block
    print('Initial Block...')
    print(f'Initial block size: {N0} samples')
    X0 = train_X[:N0]
    y0 = train_y[:N0]

    labels_set = set(y0)
    print(f'Initially available labels are: {labels_set}')
    labels2index_map = {label: ind for ind, label in enumerate(labels_set)}
    print(f'Labels to index map: {labels2index_map}')
    y0 = np.array([labels2index_map[label] for label in y0])
    y0_bipolar = to_bipolar(y0, labels2index_map)

    input_dim = X0.shape[1]
    hidden_dim = nHidden
    output_dim = y0_bipolar.shape[1]

    i2h = np.random.uniform(-1, 1, (input_dim, hidden_dim))
    print(f'Network is created with {input_dim} input neurons, {hidden_dim} hidden neurons '
          f'and {output_dim} output neurons')

    H0 = sigmoid(np.dot(X0, i2h))
    M0 = inv(np.dot(np.transpose(H0), H0))
    beta0 = np.dot(np.dot(M0, np.transpose(H0)), y0_bipolar)
    H = H0
    M = M0
    beta = beta0

    ##Subsequent Block
    print(f'Learning {train_X[N0:].shape[0]} samples sequentially with mini batch size: {batch_size}')
    print('Learning Sequentially...')
    for X, y in get_batch_data_train(batch_size, train_X[N0:], train_y[N0:]):
        X = X.reshape((batch_size, 4))
        y = y.reshape((batch_size)).tolist()
        batch_labels = set(y)
        new_labels = batch_labels - labels_set
        c = len(new_labels)
        if c:
            print(f'{c} new labels have appeared\nNewly introduced labels are {new_labels}')
            labels_set = labels_set | new_labels
            print(f'Added new labels to the labels set\nRevised labels set is {labels_set}')
            for label in new_labels:
                labels2index_map[label] = len(labels2index_map)
            print(f'Revised labels to index map: {labels2index_map}')
            print('Continuing to learn...')

            N_prime = nHidden
            m = len(labels_set) - c
            b = H.shape[0] #Previous batch size

            beta_tilde = np.dot(beta, np.eye(m, m+c))
            Nr1 = np.dot(M, np.transpose(H))
            Nr2 = -1 * np.ones((b, c))
            delta_beta_c = np.dot(Nr1, Nr2)
            delta_beta_mc = np.concatenate((np.zeros((N_prime, m)), delta_beta_c), axis=1)
            beta_mc = beta_tilde + delta_beta_mc
            beta = beta_mc



        y = np.array([labels2index_map[label] for label in y])
        y_bipolar = to_bipolar(y, labels2index_map)

        H = sigmoid(np.dot(X, i2h))
        Dr = np.eye(H.shape[0]) + np.dot(np.dot(H, M), np.transpose(H))
        Nr1 = np.dot(M, np.transpose(H))
        Nr2 = inv(Dr)
        Nr = np.dot(np.dot(np.dot(Nr1, Nr2), H), M)
        M = M - Nr
        Nr3 = y_bipolar - np.dot(H, beta)
        Nr4 = np.dot(np.dot(M, np.transpose(H)), Nr3)
        beta = beta + Nr4

    print('Training Done...')
    print('Begin Testing...')
    #ELM Testing
    H = sigmoid(np.dot(test_X, i2h))
    y_pred = np.dot(H, beta)
    y_pred = np.array([np.argmax(y_pred[i, :]) for i in range(len(y_pred))])

    # print(y_pred)
    test_y = np.array([labels2index_map[str(label)] for label in test_y])
    # print(test_y)

    print('Generating Classification Report...')
    print(classification_report(test_y, y_pred))
    print(f'Accuracy Score: {accuracy_score(test_y, y_pred)}')

    pass