import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold, train_test_split
import cPickle as pickle
from scipy.optimize import minimize

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

def mae_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, hold):
            final_prediction += weight*hold[prediction]

    return mean_absolute_error(y_hold, final_prediction)

starting_values = np.random.uniform(size=hold.shape[1])

#cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
cons = ({'type':'ineq', 'fun': lambda w: w})
bounds = [(0,1)]*hold.shape[1]

res = minimize(mae_loss_func, 
           starting_values, 
           method = 'SLSQP' 
           bounds = bounds, 
           constraints = cons,
           options={'maxiter': 100000})

best_score = res['fun']
weights = res['x']

print('Ensamble Score: {}'.format(best_score))
print('Best Weights: {}'.format(weights))

pred_hold = 0
for weight, prediction in zip(weights, hold):
        pred_hold += weight*hold[prediction]    

hold_mae = mean_absolute_error(np.exp(y_hold)-200, np.exp(pred_hold)-200)
#1137.6201662390527


pred_test = 0
for weight, prediction in zip(weights, test):
        pred_test += weight*test[prediction]


## test predictions
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(path + 'submission/stack_weight_test' + str(round(hold_mae,6)) + '.csv', index = False)

