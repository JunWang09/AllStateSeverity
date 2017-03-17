import os
mingw_path = 'C:/Program Files/mingw-w64/x86_64-6.2.0-posix-seh-rt_v5-rev1/mingw64/bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold, train_test_split
import cPickle as pickle


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))
    
RANDOM_STATE = 6688
params = {
    'min_child_weight': 200,
    'eta': 0.01,
    'colsample_bytree': 0.5,
    'max_depth': 18,
    'subsample': 0.8,
    'alpha': 1,
    'gamma': 1,
    'silent': 1,
    'seed': RANDOM_STATE
}

path = 'C:/Users/junwan/Desktop/Projects/K/all_severity/'

train = pd.read_csv(path + 'data/train.csv')
test = pd.read_csv(path + 'data/test.csv')
split = pd.read_csv(path + 'data/split_ind.csv')

train.shape, test.shape, split.shape

train_tvh = pd.merge(train, split, how='inner', on=['id'])
train_tv = train_tvh[train_tvh['split']==0].reset_index(drop=True)
id_train_tv = train_tv['id']
y_train_tv = np.log(train_tv['loss'] + 200)

hold = train_tvh[train_tvh['split']==1].reset_index(drop=True)
id_hold = hold['id']
y_hold = np.log(hold['loss'] + 200)
    
id_test = test['id']
test['loss'] = np.nan
test['split'] = 2

train_tvh.shape, train_tv.shape, hold.shape, test.shape, id_train_tv.shape, id_hold.shape, id_test.shape

pred_oob = np.zeros(train_tv.shape[0])
pred_hold = np.zeros(hold.shape[0])
pred_test = np.zeros(test.shape[0])

nfolds = 10
folds = KFold(train_tv.shape[0], n_folds= nfolds, shuffle=True, random_state=111)

for i, (t_index, v_index) in enumerate(folds):
    print('\n Fold %d' % (i+1))
   
    train_t, train_v = train_tv.ix[t_index], train_tv.ix[v_index]
    
    train_v['split'] = 3
    tv = train_t.append(train_v)
    tvh = tv.append(hold)
    train_test = tvh.append(test)
    
    cats = train_test.dtypes[train_test.dtypes=='object'].index
    for var in cats:
        train_test[var] = loo_1(train_test, [var], train_test.loss, train_test.split)
        
    split = train_test.split
    y_train = np.log(train_test['loss'][split==0] + 200)
    y_val = np.log(train_test['loss'][split==3] + 200)
    
    train_test.drop(['loss','id','split'],axis=1,inplace=True)
    
    x_train = train_test[split==0]
    x_val = train_test[split==3]
    x_hold = train_test[split==1]
    x_test = train_test[split==2]
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dhold = xgb.DMatrix(x_hold, label=y_hold)
    dtest = xgb.DMatrix(x_test)
    watchlist = [(dtrain, 'train'), (dhold, 'hold')]      
    clf = xgb.train(params, dtrain, 100000, watchlist, early_stopping_rounds=500, verbose_eval=10, feval=evalerror)
    
    pred_oob[v_index] = np.exp(clf.predict(dval, ntree_limit=clf.best_ntree_limit)) - 200 
    pred_hold += np.exp(clf.predict(dhold, ntree_limit=clf.best_ntree_limit)) - 200
    pred_test += np.exp(clf.predict(dtest, ntree_limit=clf.best_ntree_limit)) - 200
    
    
## train predictions
cv_mae = mean_absolute_error(np.exp(y_train_tv)-200, pred_oob)
print('Total - CV MAE:', cv_mae)
#('Total - CV MAE:', 1132.4347902586055)

df = pd.DataFrame({'id': id_train_tv, 'loss': pred_oob})
df.to_csv(path + 'cache/stack_xgb_loo_oob.csv', index = False)

## hold predictions
pred_hold /= nfolds
hold_mae = mean_absolute_error(np.exp(y_hold)-200, pred_hold)
print('Total - Hold MAE:', hold_mae)
#('Total - Hold MAE:', 1141.2920924640612)

df = pd.DataFrame({'id': id_hold, 'loss': pred_hold})
df.to_csv(path + 'cache/stack_xgb_loo_hold.csv', index = False)

## test predictions
pred_test /= nfolds
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(path + 'cache/stack_xgb_loo_test.csv', index = False)







