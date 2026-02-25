import xgboost
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn import metrics
import sys

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

def mkd(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

day = '20-10'
fc_ctrl = 1

def exec(typ):
    ldata, name = lpdb_data, lpdb

    folders = [f'results',
               f'results/xgboost_{day}',
               f'results/xgboost_{day}/PDB',
               f'results/xgboost_{day}/PDB/eval',
               f'results/xgboost_{day}/PDB/it']
    for f in folders:
        mkd(f)

    for d in [0]:
        print('FC-', d + fc_ctrl)

        data = ldata[d]
        x = np.array(data.iloc[:, :-1])
        y = np.array(data.iloc[:, -1])

        data_it = shuffle(ldata[d + int(len(ldata)/2)], random_state=111)
        pd.DataFrame(data_it.index).to_csv(f'results/xgboost_{day}/{typ}/index_it.csv')

        x_it = np.array(data_it.iloc[:, :-1])
        y_it = np.array(data_it.iloc[:, -1])

        acc, f1 = [], []

        kf = KFold(n_splits=10, shuffle=True, random_state=111)
        index_tr = pd.DataFrame([])
        index_ts = pd.DataFrame([])
        c = 1
        for train, test in kf.split(x, y):
            index_tr = pd.concat([index_tr, pd.DataFrame(train)], axis=1)
            index_ts = pd.concat([index_ts, pd.DataFrame(test)], axis=1)

            clf = xgboost.XGBClassifier(
                colsample_bylevel= 0.8506899781724641, colsample_bytree= 0.17219010848522695, gamma= 1.675823794985098e-05, learning_rate= 0.028014182714153124, max_delta_step= 0, max_depth= 16, min_child_weight= 0.7059430081342065, n_estimators= 194, reg_alpha= 1.2905368590789031e-07, reg_lambda= 0.009433462784709972, scale_pos_weight= 0.6177491952898517, subsample= 0.857344708916481, tree_method= 'approx' #1
                #colsample_bylevel= 0.2, colsample_bytree= 0.9504230806447873, gamma= 6.964110402415467e-05, learning_rate= 0.09440117919133446, max_delta_step= 19, max_depth= 10, min_child_weight= 1.4213089219872137, n_estimators= 185, reg_alpha= 1.366354168505398e-06, reg_lambda= 0.0004634492281691692, scale_pos_weight= 0.7065131611930926, subsample= 0.8774566112271143, tree_method= 'hist' #2
                #colsample_bylevel= 0.8571776924171046, colsample_bytree= 0.9617455291900485, gamma= 0.038054991622747364, learning_rate= 0.06261617692921269, max_delta_step= 1, max_depth= 4, min_child_weight= 0.181016026397066, n_estimators= 380, reg_alpha= 5.951660631505972e-07, reg_lambda= 3.300012890326377e-08, scale_pos_weight= 0.17895820140127336, subsample= 0.9934995154106276, tree_method= 'auto' #3
                #colsample_bylevel= 0.35, colsample_bytree= 0.8504230806447873, gamma= 6.964110402415467e-05, learning_rate= 0.08440117919133446, max_delta_step= 19, max_depth= 16, min_child_weight= 1.4213089219872137, n_estimators= 185, reg_alpha= 1.366354168505398e-06, reg_lambda= 0.0004634492281691692, scale_pos_weight= 0.7065131611930926, subsample= 0.8774566112271143, tree_method= 'hist' #4
            )

            hist = clf.fit(x[train], y[train])

            prob = clf.predict_proba(x[test])
            pred = np.argmax(prob, axis=1)
            pred = pd.concat([pd.DataFrame(y[test]), pd.DataFrame(pred)], axis=1)
            pred.columns = ['y', 'pred']

            acc.append(metrics.accuracy_score(pred['y'], pred['pred']))
            f1.append(metrics.f1_score(pred['y'], pred['pred']))

            pd.DataFrame(prob).to_csv(f'results/xgboost_{day}/{typ}/eval/prob_{c}_FC-{d + fc_ctrl}.csv')
            pred.to_csv(f'results/xgboost_{day}/{typ}/eval/pred_{c}_FC-{d + fc_ctrl}.csv')

            c += 1

            index_tr.to_csv(f'results/xgboost_{day}/{typ}/index_tr_FC{d + fc_ctrl}.csv')
            index_ts.to_csv(f'results/xgboost_{day}/{typ}/index_ts_FC{d + fc_ctrl}.csv')

        # ---------------- it ----------------
        clf_ = xgboost.XGBClassifier(
            colsample_bylevel=0.8506899781724641, colsample_bytree=0.17219010848522695, gamma=1.675823794985098e-05, learning_rate=0.028014182714153124, max_delta_step=0, max_depth=16, min_child_weight=0.7059430081342065, n_estimators=194, reg_alpha=1.2905368590789031e-07, reg_lambda=0.009433462784709972, scale_pos_weight=0.6177491952898517, subsample=0.857344708916481, tree_method='approx'  # 1
            #colsample_bylevel= 0.2, colsample_bytree= 0.9504230806447873, gamma= 6.964110402415467e-05, learning_rate= 0.09440117919133446, max_delta_step= 19, max_depth= 10, min_child_weight= 1.4213089219872137, n_estimators= 185, reg_alpha= 1.366354168505398e-06, reg_lambda= 0.0004634492281691692, scale_pos_weight= 0.7065131611930926, subsample= 0.8774566112271143, tree_method= 'hist' #2
            #colsample_bylevel= 0.8571776924171046, colsample_bytree= 0.9617455291900485, gamma= 0.038054991622747364, learning_rate= 0.06261617692921269, max_delta_step= 1, max_depth= 4, min_child_weight= 0.181016026397066, n_estimators= 380, reg_alpha= 5.951660631505972e-07, reg_lambda= 3.300012890326377e-08, scale_pos_weight= 0.17895820140127336, subsample= 0.9934995154106276, tree_method= 'auto' #3
            #colsample_bylevel= 0.35, colsample_bytree= 0.8504230806447873, gamma= 6.964110402415467e-05, learning_rate= 0.08440117919133446, max_delta_step= 19, max_depth= 16, min_child_weight= 1.4213089219872137, n_estimators= 185, reg_alpha= 1.366354168505398e-06, reg_lambda= 0.0004634492281691692, scale_pos_weight= 0.7065131611930926, subsample= 0.8774566112271143, tree_method= 'hist' #4
        )
        hist = clf_.fit(x, y)

        prob_it = clf_.predict_proba(x_it)

        pred_it = np.argmax(prob_it, axis=1)
        pred_it = pd.concat([pd.DataFrame(y_it), pd.DataFrame(pred_it)], axis=1)
        pred_it.columns = ['y', 'pred']

        acc_ = metrics.accuracy_score(pred_it['y'], pred_it['pred'])
        f1_ = metrics.f1_score(pred_it['y'], pred_it['pred'])

        pd.DataFrame(prob_it).to_csv(f'results/xgboost_{day}/{typ}/it/prob_11_FC-{d + fc_ctrl}.csv')
        pd.DataFrame(pred_it).to_csv(f'results/xgboost_{day}/{typ}/it/pred_11_FC-{d + fc_ctrl}.csv')

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
