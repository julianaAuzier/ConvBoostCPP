import pandas as pd
from sklearn import metrics
import numpy as np
import os

pd.options.display.max_rows = 999

day = '20-10'

def mkd(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def voting(prob, actual, i, k):
    probs = pd.concat(prob, axis=1)
    c0 = probs['0'].mean(axis=1)
    c1 = probs['1'].mean(axis=1)

    actual = np.array(actual)

    final_pred = np.argmax([c0, c1], axis=0)

    df = pd.concat([pd.DataFrame(actual),pd.DataFrame(final_pred)], axis=1)
    df_ = pd.concat([pd.DataFrame(actual),pd.concat([c0,c1], axis=1)], axis=1)

    df.columns = ['y','pred']
    df_.columns = ['y','prob_0','prob_1']

    df.to_csv(fr'results\voting_{day}\PDB\eval\pred_{i}_FC-{k}.csv')
    df_.to_csv(fr'results\voting_{day}\PDB\eval\prob_{i}_FC-{k}.csv')

    results = []

    acc = metrics.accuracy_score(actual, final_pred)
    mcc = metrics.matthews_corrcoef(actual, final_pred)
    pre = metrics.precision_score(actual, final_pred)
    rec = metrics.recall_score(actual, final_pred)
    f1 = metrics.f1_score(actual, final_pred)
    fpr, tpr, thresholds = metrics.roc_curve(actual, final_pred)
    auc = metrics.auc(fpr, tpr)
    sp = metrics.recall_score(actual, final_pred, pos_label=0)

    results.append([acc, pre, rec, mcc, f1, sp, auc])

    return final_pred, results

#--------------------------------------------------------------
#pathPDB = os.getcwd() + r'..\..\data\unmodified\PDB' + '\\'
pathPDB = os.getcwd() + r'\data\PDB' + '\\'
lpdb = os.listdir(pathPDB)
lpdb_data = [pd.read_csv(pathPDB + i, index_col=0) for i in lpdb]
#--------------------------------------------------------------
folders = [f'results',
           f'results/voting_{day}',
           f'results/voting_{day}/PDB',
           f'results/voting_{day}/PDB/eval',
           f'results/voting_{day}/PDB/it']
for f in folders:
    mkd(f)
#--------------------------------------------------------------
print('pdb-cv')
for k in [1,2,3,4]:
    metrics_cv = pd.DataFrame([])
    for i in range(1,11):
        cnn = pd.read_csv(fr'{os.getcwd()}\results\imcnn_{day}\PDB\eval\prob_{i}_FC-{k}.csv', index_col=0)
        xgb = pd.read_csv(fr'{os.getcwd()}\results\xgboost_{day}\PDB\eval\prob_{i}_FC-{k}.csv', index_col=0)
        idx = pd.read_csv(fr'{os.getcwd()}\results\imcnn_{day}\PDB\index_ts_FC{k}.csv', index_col=0)
        idx_ = pd.read_csv(fr'{os.getcwd()}\results\xgboost_{day}\PDB\index_ts_FC{k}.csv', index_col=0)

        y = lpdb_data[k-1].iloc[:,-1]
        idx.columns = list(range(1,11))

        index = idx[i]

        act = y[index]

        final = voting([cnn, xgb], act, i, k)

        metrics_cv= pd.concat([metrics_cv,pd.DataFrame(final[1])], axis=0, ignore_index=True)

    metrics_cv.columns = ['acc','pre','rec','mcc', 'f1', 'sp', 'auc']

    m_acc = metrics_cv['acc'].mean(axis=0)
    metrics_cv.to_csv(fr'{os.getcwd()}\results\voting_{day}\PDB\eval_FC{k}.csv')
    print(metrics_cv)
    print('acc: ', m_acc)
