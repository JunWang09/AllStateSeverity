
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
    num_iterations=10000,
    learning_rate=0.01,
    num_leaves=200,
    num_threads=4,
    min_sum_hessian_in_leaf=1,
    metric='l1',
    feature_fraction=0.5,
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

test['loss'] = np.nan
test['split'] = 2
train_test = train_tvh.append(test)

train_tvh.shape, test.shape, train_test.shape

for column in list(train_tvh.select_dtypes(include=['object']).columns):
    if train_tvh[column].nunique() != test[column].nunique():
        set_train = set(train_tvh[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)
        def filter_cat(x):
            if x in remove:
                return np.nan
            return x

        train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)
        
    train_test[column] = pd.factorize(train_test[column].values, sort=True)[0]

train_tv = train_test[train_test['split']==0].reset_index(drop=True)
hold = train_test[train_test['split']==1].reset_index(drop=True)
test = train_test[train_test['split']==2]

id_train_tv = train_tv['id']
id_hold = hold['id']
id_test = test['id']

y_train_tv = np.log(train_tv['loss'] + 200)
y_hold = np.log(hold['loss'] + 200)

x_train_tv = train_tv.drop(['loss', 'id', 'split'], 1)
x_hold = hold.drop(['loss', 'id', 'split'], 1)
x_test = test.drop(['loss', 'id', 'split'], 1)

pred_oob = np.zeros(train_tv.shape[0])
pred_hold = np.zeros(hold.shape[0])
pred_test = np.zeros(test.shape[0])

nfolds = 2
folds = KFold(x_train_tv.shape[0], n_folds=nfolds, shuffle=True, random_state=111)

for i, (t_index, v_index) in enumerate(folds):
    x_train, x_val = x_train_tv.ix[t_index], x_train_tv.ix[v_index]
    y_train, y_val = y_train_tv.ix[t_index], y_train_tv.ix[v_index]
         
    clf.fit(x_train, y_train, test_data=[(x_hold, y_hold)])   
    pred_oob[v_index] = np.exp(clf.predict(x_val)) - 200 
    pred_hold += np.exp(clf.predict(x_hold)) - 200
    pred_test += np.exp(clf.predict(x_test)) - 200

## train predictions
cv_mae = mean_absolute_error(np.exp(y_train_tv)-200, pred_oob)
print('Total - CV MAE:', cv_mae)
#('Total - CV MAE:', 1144.1571654045865)

df = pd.DataFrame({'id': id_train_tv, 'loss': pred_oob})
df.to_csv(path + 'cache/stack_lgbm_fctd_oob_1.csv', index = False)

## hold predictions
pred_hold /= nfolds
hold_mae = mean_absolute_error(np.exp(y_hold)-200, pred_hold)
print('Total - Hold MAE:', hold_mae)
#('Total - Hold MAE:', 1143.8217623792966)

df = pd.DataFrame({'id': id_hold, 'loss': pred_hold})
df.to_csv(path + 'cache/stack_lgbm_fctd_hold_1.csv', index = False)

## test predictions
pred_test /= nfolds
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(path + 'cache/stack_lgbm_fctd_test_1.csv', index = False)



