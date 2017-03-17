
import numpy as np
import pandas as pd
from pylightgbm.models import GBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
import itertools

exec_path = "C:/LightGBM-master/windows/x64/Release/lightgbm.exe"

COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(',')

clf = GBMRegressor(
    exec_path=exec_path,
    config='',
    application='regression',
    num_iterations=100000,
    learning_rate=0.01,
    #num_leaves=200,
    num_threads=4,
    min_sum_hessian_in_leaf=500,
    metric='l1',
    feature_fraction=0.3,
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

numeric_feats = [x for x in train_tvh.columns[1:-1] if 'cont' in x]

train_test = mungeskewed(train_tvh, test, numeric_feats)

for comb in itertools.combinations(COMB_FEATURE, 2):
    feat = comb[0] + "_" + comb[1]
    train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
    train_test[feat] = train_test[feat].apply(encode)
    print(feat)

cats = [x for x in train_tvh.columns[1:-1] if 'cat' in x]
for col in cats:
    train_test[col] = train_test[col].apply(encode)

ss = StandardScaler()
train_test[numeric_feats] = ss.fit_transform(train_test[numeric_feats].values)

train_tv = train_test[train_test['split']==0].reset_index(drop=True)
hold = train_test[train_test['split']==1].reset_index(drop=True)
test = train_test[train_test['split']==2]

id_train_tv = train_tv['id']
id_hold = hold['id']
id_test = test['id']

y_train_tv = np.log(train_tv['loss'] + 200)
y_hold = np.log(hold['loss'] + 200)
y_test = np.log(test['loss'] + 200)

x_train_tv = train_tv.drop(['loss', 'id', 'split'], 1)
x_hold = hold.drop(['loss', 'id', 'split'], 1)
x_test = test.drop(['loss', 'id', 'split'], 1)

pred_oob = np.zeros(train_tv.shape[0])
pred_hold = np.zeros(hold.shape[0])
pred_test = np.zeros(test.shape[0])

nfolds = 10
folds = KFold(x_train_tv.shape[0], n_folds=nfolds, shuffle=True, random_state=111)

for i, (t_index, v_index) in enumerate(folds):
    print('\n Fold {0}'.format(i + 1))
    x_train, x_val = x_train_tv.ix[t_index], x_train_tv.ix[v_index]
    y_train, y_val = y_train_tv.ix[t_index], y_train_tv.ix[v_index]


    clf.fit(x_train, y_train, test_data=[(x_hold, y_hold)])   
    pred_oob[v_index] = np.exp(clf.predict(x_val)) - 200 
    pred_hold += np.exp(clf.predict(x_hold)) - 200
    pred_test += np.exp(clf.predict(x_test)) - 200

## train predictions
cv_mae = mean_absolute_error(np.exp(y_train_tv)-200, pred_oob)
print('Total - CV MAE:', cv_mae)
#('Total - CV MAE:', 1132.1838231460306)

df = pd.DataFrame({'id': id_train_tv, 'loss': pred_oob})
df.to_csv(path + 'cache/stack_lgbm_encode_oob.csv', index = False)

## hold predictions
pred_hold /= nfolds
hold_mae = mean_absolute_error(np.exp(y_hold)-200, pred_hold)
print('Total - Hold MAE:', hold_mae)
#('Total - Hold MAE:', 1140.1594738169745)

df = pd.DataFrame({'id': id_hold, 'loss': pred_hold})
df.to_csv(path + 'cache/stack_lgbm_encode_hold.csv', index = False)

## test predictions
pred_test /= nfolds
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(path + 'cache/stack_lgbm_encode_test.csv', index = False)




