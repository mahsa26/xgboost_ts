
from numbers_parser import Document
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def convert2matrix(data_arr_on, data_arr_off , look_back, feature_idx0, feature_idx1):
    X, Y =[], []
    for i in range(len(data_arr_on)-look_back):
        d=i+look_back  
        X.append(np.concatenate((data_arr_off[i:d,feature_idx0],data_arr_off[i:d,feature_idx1])))
        Y.append(data_arr_on[d])
    return np.array(X), np.array(Y)

# np.concatenate(data_arr_on[i:d,feature_idx0],data_arr_on[i:d,feature_idx1])

def model_dnn(optimizer='adam', activation_hl01='relu', activation_hl02='relu', num_node01=32, num_node02=8):
    model = Sequential()
    model.add(Dense(units=num_node01, input_dim=look_back*num_features, activation=activation_hl01))
    model.add(Dense(num_node02, activation=activation_hl02))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    return model


def model_loss(history):
    plt.figure(figsize=(16,8))# 16 is x-axis length, 4 is y-axis length
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show()

# Read file
doc = Document("offshore.numbers")
sheets = doc.sheets()
#for i in range(len(sheets)):
    #print(sheets[i].name)
print(len(sheets))
for i in range(len(sheets)):
    tables = sheets[i].tables()#; print('tables', tables)
    data_off = np.array(tables[0].rows(values_only=True))
    if i%4 == 0:
        columns = data_off[0]
        data_off = data_off[458:,:]; print('i-{} rows-{}'.format(i, np.shape(data_off)[0]))
        if i==0:
            all_data_off = np.copy(data_off)
        else:
            all_data_off = np.concatenate((all_data_off, data_off))
    elif i%4 == 3:
        data_off = data_off[1:482,:]; print('i-{} rows-{}'.format(i, np.shape(data_off)[0]))
        all_data_off = np.concatenate((all_data_off, data_off))
    else:
        data_off = data_off[1:,:]; print('i-{} rows-{}'.format(i, np.shape(data_off)[0]))
        all_data_off = np.concatenate((all_data_off, data_off))

print('all_data (offshore) ', np.shape(all_data_off))

label_feature_idx = 2
doc = Document("nearshore.numbers")
sheets = doc.sheets()
#for i in range(len(sheets)):
    #print(sheets[i].name)
print(len(sheets))
for i in range(len(sheets)):
    tables = sheets[i].tables()#; print('tables', tables)
    data_on = np.array(tables[0].rows(values_only=True))
    if i%4 == 0:
        data_on = data_on[458:,label_feature_idx]; print('i-{} rows-{}'.format(i, np.shape(data_on)[0]))
        if i==0:
            all_data_on = np.copy(data_on)
        else:
            all_data_on = np.concatenate((all_data_on, data_on))
    elif i%4 == 3:
        data_on = data_on[1:482,label_feature_idx]; print('i-{} rows-{}'.format(i, np.shape(data_on)[0]))
        all_data_on = np.concatenate((all_data_on, data_on))
    else:
        data_on = data_on[1:,label_feature_idx]; print('i-{} rows-{}'.format(i, np.shape(data_on)[0]))
        all_data_on = np.concatenate((all_data_on, data_on))

print('all_data (nearshore) ', np.shape(all_data_on))


#print(all_data)

# approximately 70% data in training set and remaining in test set
train_size = int(.7*np.shape(all_data_off)[0]); print(train_size, len(all_data_off))
# training data without labels: 2-D
train_off, test_off= all_data_off[0:train_size, :], all_data_off[train_size:len(all_data_off)+1, :]
# label data: 1-D
train_on, test_on = all_data_on[0:train_size], all_data_on[train_size:len(all_data_on)+1]

print(train_off.shape, test_off.shape, train_on.shape, test_on.shape)

# Prepare time series dataset
num_features=2
look_back = 2
#0:Day, 1:hour, 2:Hs, 3:spr, 4:dir, 5:Tm, 6:Tp
feature_idx0 = 2
feature_idx1 = 3
trainX, trainY = convert2matrix(train_on, train_off, look_back, feature_idx0, feature_idx1)
testX, testY = convert2matrix(test_on, train_off, look_back, feature_idx0, feature_idx1)
print(np.shape(trainX), np.shape(trainY), np.shape(testX), np.shape(testY))

trainX = np.asarray(trainX).astype(np.float32)
testX = np.asarray(testX).astype(np.float32)
trainY = np.asarray(trainY).astype(np.float32)
testY = np.asarray(testY).astype(np.float32)

xgb = xgboost.XGBRegressor(n_estimators = 10, tree_method="hist", eval_metric=mean_absolute_error)
xgb.fit(trainX, trainY, eval_set=[(testX, testY)])
y_pred = xgb.predict(testX)

print('Mean Absolute Error:', metrics.mean_absolute_error(testY, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(testY, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY, y_pred)))


'''
#Grid Search
batch_size = [8]#,16,32,64,128]#power of 2
epochs = [10]#50,100,500,1000]
optimizer = ['adam']#,'RMSprop']#, 'SGD']#, 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
activation_hl01 = ['relu']#, 'tanh']#, 'softmax', 'softplus', 'softsign', 'sigmoid', 'hard_sigmoid', 'linear']
activation_hl02 = ['relu']#, 'tanh']#, 'softmax', 'softplus', 'softsign', 'sigmoid', 'hard_sigmoid', 'linear']
num_node01 = [2]#,3,4, 8, 32]
num_node02 = [2]#,3,4, 8, 32]
model = KerasClassifier(build_fn=model_dnn, verbose=0)
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, activation_hl01=activation_hl01, activation_hl02=activation_hl02, num_node01=num_node01, num_node02=num_node02)

#define your own mse and set greater_is_better=False
mse = make_scorer(mean_squared_error, greater_is_better=False)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=0, scoring=mse)
grid_result = grid.fit(trainX, trainY)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
    #print("%f (%f) with: %r" % (mean, stdev, param))
#print(grid.predict(testX))


# model.evaluate returns the loss value & metrics values for the model
train_score = model.evaluate(trainX, trainY, verbose=0)
print("Mean squared error (train)", train_score)
#print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '% (np.sqrt(train_score[1]), train_score[2]))
test_score = model.evaluate(testX, testY, verbose=0)
print("Mean squared error (test)", test_score)
#print('Test Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '% (np.sqrt(test_score[1]), test_score[2]))
test_predict = model.predict(testX)
'''
def cal_cc(testY, test_predict):
    part01 = testY - np.mean(testY)
    part02 = test_predict - np.mean(test_predict)
    num = np.sum(part01 * part02)
    den = np.sum(part01**2) * np.sum(part02**2)
    if den == 0:
        return 0
    return num/np.sqrt(den)
print("Pearson Correlation Coefficient (test)", cal_cc(testY, y_pred))
