
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold, train_test_split
import cPickle as pickle
from pylightgbm.models import GBMRegressor

exec_path = "C:/LightGBM-master/windows/x64/Release/lightgbm.exe"

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))
    
clf = GBMRegressor(
    exec_path=exec_path,
    config='',
    application='regression',
    num_iterations=100000,
    learning_rate=0.01,
    #num_leaves=200,
    num_threads=4,
    min_sum_hessian_in_leaf=200,
    metric='l1',
    feature_fraction=0.4,
    feature_fraction_seed=42,
    bagging_fraction=0.8,
    bagging_freq=100,
    bagging_seed=42,
    early_stopping_round=500,
    verbose=True
   )

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
    
    clf.fit(x_train, y_train, test_data=[(x_hold, y_hold)]) 

    pred_oob[v_index] = np.exp(clf.predict(x_val)) - 200 
    pred_hold += np.exp(clf.predict(x_hold)) - 200
    pred_test += np.exp(clf.predict(x_test)) - 200
 
    
## train predictions
cv_mae = mean_absolute_error(np.exp(y_train_tv)-200, pred_oob)
print('Total - CV MAE:', cv_mae)
#('Total - CV MAE:', 1134.2307373725141)

df = pd.DataFrame({'id': id_train_tv, 'loss': pred_oob})
df.to_csv(path + 'cache/stack_lgbm_loo_oob_1.csv', index = False)

## hold predictions
pred_hold /= nfolds
hold_mae = mean_absolute_error(np.exp(y_hold)-200, pred_hold)
print('Total - Hold MAE:', hold_mae)
#('Total - Hold MAE:', 1141.5409108889596)

df = pd.DataFrame({'id': id_hold, 'loss': pred_hold})
df.to_csv(path + 'cache/stack_lgbm_loo_hold_1.csv', index = False)

## test predictions
pred_test /= nfolds
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(path + 'cache/stack_lgbm_loo_test_1.csv', index = False)

df.loss.describe()






