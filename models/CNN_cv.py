import os
from keras import utils
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Dropout, LeakyReLU, Dense, Flatten, AveragePooling1D
from keras.regularizers import L1L2, L1, L2
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sys

def mkd(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def imCNN():
    model_imCNN = Sequential()

    model_imCNN.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='tanh'))
    model_imCNN.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='tanh'))

    model_imCNN.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='tanh',
                           activity_regularizer=L1(1e-3)
                           ))
    model_imCNN.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='tanh',
                           activity_regularizer=L1(1e-3)
                           ))

    model_imCNN.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='tanh',
                           bias_regularizer=L1(1e-2),
                           activity_regularizer=L1(1e-3)
                           ))
    model_imCNN.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='tanh',
                           bias_regularizer=L1(1e-2),
                           activity_regularizer=L1(1e-3)
                           ))
    model_imCNN.add(MaxPooling1D(pool_size=2))

    model_imCNN.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='tanh',
                           bias_regularizer=L1(1e-2),
                           activity_regularizer=L1(1e-3)
                           ))
    model_imCNN.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='tanh',
                           bias_regularizer=L1(1e-2),
                           activity_regularizer=L1(1e-3)))
    model_imCNN.add(MaxPooling1D(pool_size=3))

    model_imCNN.add(Dropout(0.5))
    model_imCNN.add(Flatten())

    model_imCNN.add(Dense(256, activation='tanh',
                          bias_regularizer=L1(1e-3),
                          activity_regularizer=L1(1e-3)))

    model_imCNN.add(Dense(128, activation='tanh',
                          bias_regularizer=L1(1e-3),
                          activity_regularizer=L1(1e-3)))

    model_imCNN.add(Dense(64, activation='tanh',
                          bias_regularizer=L1(1e-3),
                          activity_regularizer=L1(1e-3)))
    model_imCNN.add(Dense(32, activation='tanh',
                          activity_regularizer=L1(1e-3)))
    model_imCNN.add(Dense(16, activation='tanh'))
    model_imCNN.add(Dense(2, activation='softmax'))
    model_imCNN.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model_imCNN

day = '20-10'
fc_ctrl = 1

def exec(typ):
    ldata, name = lpdb_data, lpdb

    folders = [f'results',
               f'results/imcnn_{day}',
               f'results/imcnn_{day}/PDB',
               f'results/imcnn_{day}/PDB/eval',
               f'results/imcnn_{day}/PDB/it']
    for f in folders:
        mkd(f)

    for d in [0,2]:
        print('FC-', d + fc_ctrl, f'| {typ}')

        data = ldata[d]

        x = np.array(data.iloc[:, :-1])
        print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = np.array(data.iloc[:, -1])

        data_it = shuffle(ldata[d + int(len(ldata)/2)], random_state=111)
        pd.DataFrame(data_it.index).to_csv(f'results/imcnn_{day}/{typ}/index_it.csv')

        x_it = np.array(data_it.iloc[:, :-1])
        x_it = x_it.reshape(x_it.shape[0], x_it.shape[1], -1)
        y_it = np.array(data_it.iloc[:, -1])

        acc, f1 = [], []

        kf = KFold(n_splits=10, shuffle=True, random_state=111)
        index_tr = pd.DataFrame([])
        index_ts = pd.DataFrame([])

        c = 1
        for train, test in kf.split(x, y):
            index_tr = pd.concat([index_tr, pd.DataFrame(train)], axis=1)
            index_ts = pd.concat([index_ts, pd.DataFrame(test)], axis=1)

            y_train = utils.to_categorical(y[train])
            clf = imCNN()
            bs = int(len(x[train]) * .1)

            hist = clf.fit(x[train],
                           y_train,
                           epochs=25, verbose=1,
                           batch_size=bs, use_multiprocessing=True)

            prob = clf.predict(x[test], verbose=False, batch_size=bs)
            pred = np.argmax(prob, axis=1)
            pred = pd.concat([pd.DataFrame(y[test]), pd.DataFrame(pred)], axis=1)
            pred.columns = ['y', 'pred']

            acc.append(metrics.accuracy_score(pred['y'], pred['pred']))
            f1.append(metrics.f1_score(pred['y'], pred['pred']))

            pd.DataFrame(prob).to_csv(f'results/imcnn_{day}/{typ}/eval/prob_{c}_FC-{d + fc_ctrl}.csv')
            pred.to_csv(f'results/imcnn_{day}/{typ}/eval/pred_{c}_FC-{d + fc_ctrl}.csv')

            c += 1

            index_tr.to_csv(f'results/imcnn_{day}/{typ}/index_tr_FC{d + fc_ctrl}.csv')
            index_ts.to_csv(f'results/imcnn_{day}/{typ}/index_ts_FC{d + fc_ctrl}.csv')

        # ---------------- it ----------------
        y_ = utils.to_categorical(y)
        clf_ = imCNN()
        bs_ = int(len(x) * .1)

        hist = clf_.fit(x,
                        y_,
                        epochs=25, verbose=1,
                        batch_size=bs_, use_multiprocessing=True)

        prob_it = clf_.predict(x_it, verbose=False, batch_size=bs_)
        pred_it = np.argmax(prob_it, axis=1)
        pred_it = pd.concat([pd.DataFrame(y_it), pd.DataFrame(pred_it)], axis=1)
        pred_it.columns = ['y', 'pred']

        acc_ = metrics.accuracy_score(pred_it['y'], pred_it['pred'])
        f1_ = metrics.f1_score(pred_it['y'], pred_it['pred'])

        pd.DataFrame(prob_it).to_csv(f'results/imcnn_{day}/{typ}/it/prob_11_FC-{d + fc_ctrl}.csv')
        pd.DataFrame(pred_it).to_csv(f'results/imcnn_{day}/{typ}/it/pred_11_FC-{d + fc_ctrl}.csv')

        print('--cv acc: ', np.mean(acc))
        print('--cv f1: ', np.mean(f1))
        print('--it acc: ', acc_)
        print('--it f1: ', f1_)


if __name__ == '__main__':
    pathPDB = os.getcwd() + r'\data\PDB' + '\\'
    # pathPDB = os.getcwd() + r'..\..\data\unmodified\PDB' + '\\'

    lpdb = os.listdir(pathPDB)
    lpdb_data = [pd.read_csv(pathPDB + i, index_col=0) for i in lpdb]

    exec('PDB')

    sys.modules[__name__].__dict__.clear()
