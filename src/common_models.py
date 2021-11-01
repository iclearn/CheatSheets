# common imports
import statsmodels.api as sm

from keras import backend as K
import keras

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def polynomial_regression(train, valid, features, target='target'):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    model = LinearRegression()
    poly = PolynomialFeatures(degree=3)
    train_x = poly.fit_transform(X=train[features])
    model.fit(train_x, train[target])

    valid_pred = model.predict(poly.transform(valid[features]))
    rmse = root_mean_squared_error(valid[target], valid_pred)
    print(rmse)
    return model


def ols_stats_model(train, valid, features, target='target'):
    mod = sm.OLS(train[target], train[features])
    res = mod.fit()
    print(res.summary())
    valid_pred = res.predict(valid[features])
    rmse = mean_squared_error(valid.target, valid_pred, squared=False)
    print('***OLS RESULT***', rmse)
    return res, rmse

# neural network model
def nn_model(train, valid, features):
    nnet = keras.Sequential()
    nnet.add(keras.layers.Dense(64, activation='relu'))
    nnet.add(keras.layers.Dense(16, activation='relu'))
    # add dropout to overcome overfitting
    nnet.add(keras.layers.Dense(1))
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    nnet.compile(optimizer=opt, loss=root_mean_squared_error)
    # callback to stop when valid loss increasing
    nnet.fit(x=train.loc[:, features].values, y=train.target.values,
             epochs=15, batch_size=16, validation_data=(valid.loc[:,features].values, valid.loc[:, 'target'].values))

    # validation set - results
    valid_pred = nnet.predict(x=valid.loc[:, features])
    rmse = mean_squared_error(valid.target, valid_pred, squared=False)
    print('***NN-MODEL RESULT***', rmse)
    return nnet, rmse

def lgbm(train, valid, features):
    model = LGBMRegressor(random_state=42, learning_rate=5e-2).fit(X=train.loc[:, features].values, y=train.target.values,
                                                                   early_stopping_rounds=2, eval_set=(valid.loc[:,features].values, valid.loc[:, 'target'].values), eval_metric='rmse')

    valid_pred = model.predict(X=valid.loc[:, features])
    rmse = mean_squared_error(valid.target, valid_pred, squared=False)
    print('***LGBM RESULT***', rmse)
    return rmse

def nn_model_d(train, valid, features):
    nnet = keras.Sequential()
    # nnet.add(Embedding(vocab_size, 8, input_length=max_length))
    # nnet.add(Flatten())
    nnet.add(keras.layers.Dense(1024, activation='relu'))
    nnet.add(keras.layers.Dropout(0.2))
    nnet.add(keras.layers.Dense(256, activation='relu'))
    nnet.add(keras.layers.Dense(128, activation='relu'))
    nnet.add(keras.layers.Dropout(0.2))
    nnet.add(keras.layers.Dense(64, activation='relu'))
    # add dropout to overcome overfitting
    nnet.add(keras.layers.Dense(1))
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    nnet.compile(optimizer=opt, loss=root_mean_squared_error)

    # callback to stop when valid loss increasing
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=2,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )

    nnet.fit(x=train.loc[:, features].values, y=train.target.values, epochs=15, batch_size=16,
             validation_data=(valid.loc[:,features].values, valid.loc[:, 'target'].values), callbacks=[early_stop])

    # validation set results
    valid_pred = nnet.predict(x=valid.loc[:, features])
    rmse = mean_squared_error(valid.target, valid_pred, squared=False)
    print('***NN-MODEL-D RESULT***', rmse)
    #nnet.fit(x=train.loc[:, stats+['embd']].values, y=train.target.values, epochs=15, batch_size=16,
    # validation_data=(valid.loc[:,stats].values, valid.loc[:, 'target'].values))
    return rmse

def svm(train, valid, features):
    regr = make_pipeline(StandardScaler(), SVR(C=2.0, epsilon=0.01))
    regr.fit(X=train.loc[:, features].values, y=train.target.values)

    valid_pred = regr.predict(X=valid.loc[:, features])
    rmse = mean_squared_error(valid.target, valid_pred, squared=False)
    print('***SVM RESULT***', rmse)
    return regr, rmse