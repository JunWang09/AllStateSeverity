import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold, train_test_split
import cPickle as pickle

from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="./stack_ridge_loo_boxcox12.log", filemode='a', level=logging.DEBUG,
                    datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                              datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)

path = 'C:/Users/junwan/Desktop/Projects/K/all_severity/'

print(" >>>Loading data...")
table_list = ['stack_keras_3layers_logy',
'stack_keras_3layers',
'stack_lgbm_encode',
'stack_lgbm_fctd',
'stack_lgbm_loo',
'stack_lgbm_loo_logy',
'stack_xgb_encode',
'stack_xgb_fctd',
'stack_xgb_loo',
'stack_rfr_count',
'stack_rfr_loo'
]


base = 'stack_ridge_loo_boxcox12'
train = pd.read_csv(path + 'cache/' + base + '_oob.csv')
hold = pd.read_csv(path + 'cache/' + base + '_hold.csv')
test = pd.read_csv(path + 'cache/' + base + '_test.csv')

train['stack_ridge_loo_boxcox12'] = np.log(train.loss + 200)
hold['stack_ridge_loo_boxcox12'] = np.log(hold.loss + 200)
test['stack_ridge_loo_boxcox12'] = np.log(test.loss + 200)

for var in table_list:
    train[var] =np.log(pd.read_csv(path + 'cache/' + var + '_oob.csv').loss + 200)
    hold[var] =np.log(pd.read_csv(path + 'cache/' + var + '_hold.csv').loss + 200)
    test[var] =np.log(pd.read_csv(path + 'cache/' + var + '_test.csv').loss + 200)
    
id_train_tv = train['id']
id_hold = hold['id']
id_test = test['id']

y_train_tv = train['y']
y_hold = hold['y']

train.drop(['id', 'loss', 'y'], 1, inplace=True)
hold.drop(['id', 'loss', 'y'], 1, inplace=True)
test.drop(['id', 'loss'], 1, inplace=True)

print(" >>>Cross Validation...")

'''
alphas = [0.000001, 0.000005, 0.000008, 0.0000001]

ridge=Ridge(normalize=True)
mae = []
for a in alphas:
	ridge.set_params(alpha=0.000001)
	ridge.fit(train, y_train_tv)
	print('alpha is {} MAE is {}'.format(a,mean_absolute_error(np.exp(y_hold), np.exp(ridge.predict(hold)))))

pd.DataFrame(ridge.coef_, index=train.columns).to_csv(path + 'plot/stack_ridge_01.csv')
'''

ridge=Ridge(normalize=True, alpha = 0.000001) 
nfolds = 10
folds = KFold(train.shape[0], n_folds= nfolds, shuffle=True, random_state=111)

pred_oob = np.zeros(train.shape[0])
pred_hold = np.zeros(hold.shape[0])
pred_test = np.zeros(test.shape[0])

for i, (t_index, v_index) in enumerate(folds):
    print('\n Fold %d' % (i+1))
   
    x_train, x_val = train.ix[t_index], train.ix[v_index]
    y_train, y_val = y_train_tv.ix[t_index], y_train_tv.ix[v_index]

    ridge.fit(x_train, y_train)    

    pred_oob[v_index] = np.exp(ridge.predict(x_val)) - 200 
    pred_hold += np.exp(ridge.predict(hold)) - 200
    pred_test += np.exp(ridge.predict(test)) - 200
    
print(" >>>Writing results...")
## train predictions
cv_mae = mean_absolute_error(np.exp(y_train_tv)-200, pred_oob)
print('Total - CV MAE:', cv_mae)
logger.info("Total - CV MAE: %.5f",  cv_mae)
#('Total - CV MAE:', 1127.7321723585314)

df = pd.DataFrame({'id': id_train_tv, 'loss': pred_oob, 'y':y_train_tv})
df.to_csv(path + 'submission/stack_ridge_01_oob.csv', index = False)

## hold predictions
pred_hold /= nfolds
hold_mae = mean_absolute_error(np.exp(y_hold)-200, pred_hold)
print('Total - Hold MAE:', hold_mae)
logger.info("Total - Hold MAE: %.5f",  hold_mae)
#('Total - Hold MAE:', 1136.3585470447931)

df = pd.DataFrame({'id': id_hold, 'loss': pred_hold, 'y':y_hold})
df.to_csv(path + 'submission/stack_ridge_01_hold.csv', index = False)

## test predictions
pred_test /= nfolds
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(path + 'submission/stack_ridge_01_test.csv', index = False)

