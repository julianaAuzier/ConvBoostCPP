from sklearn import metrics
import pandas as pd
import os

day = '20-10'

def calc_m(y, pred):
    acc = metrics.accuracy_score(y, pred)
    mcc = metrics.matthews_corrcoef(y, pred)
    pre = metrics.precision_score(y, pred)
    rec = metrics.recall_score(y, pred) #sn
    f1 = metrics.f1_score(y, pred)
    roc_auc = metrics.roc_auc_score(y_true=y,y_score=pred)

    sp = metrics.recall_score(y,pred,pos_label=0)

    return acc, pre, rec, mcc, f1, sp, roc_auc

typ= 'PDB'
models = [f'lgbm_{day}',f'svm_{day}',f'rf_{day}',f'knn_{day}',f'mlp_{day}',f'gpc_{day}',f'imcnn_{day}',f'xgboost_{day}']
for model_result in models:
    k= 'eval'
    for fc in range(1,5):
        final_results = []
        for i in range(1,11):
            data = pd.read_csv(fr'{os.getcwd()}\results\{model_result}\{typ}\{k}\pred_{i}_FC-{fc}.csv', index_col=0)
            final_results.append(calc_m(data['y'], data['pred']))
        final_results = pd.DataFrame(final_results, columns = ['acc', 'pre', 'rec', 'mcc', 'f1', 'sp','roc_auc'])
        final_results.to_csv(fr'{os.getcwd()}\results\{model_result}\{typ}\{k}\metrics_FC-{fc}.csv')

print(100*'-')

for model_result in models:
    k = 'it'
    for fc in range(1,5):
        final_results = []
        data = pd.read_csv(fr'{os.getcwd()}\results\{model_result}\{typ}\{k}\pred_11_FC-{fc}.csv', index_col=0)
        final_results.append(calc_m(data['y'], data['pred']))
        final_results = pd.DataFrame(final_results, columns = ['acc', 'pre', 'rec', 'mcc', 'f1', 'sp','roc_auc'])
        final_results.to_csv(fr'{os.getcwd()}\results\{model_result}\{typ}\{k}\metrics_FC-{fc}.csv')
