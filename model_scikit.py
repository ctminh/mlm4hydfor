import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


# Load input X and output y
X = pd.read_csv("X.csv").to_numpy()
y = pd.read_csv("y.csv").to_numpy()
# X_test = pd.read_csv("X_test.csv")

# Preprocessing the dataset
def preprocess_standarization(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

# Define the training function
def training_model(model, X, y):
    # variance storing Exp-variance, R2, MSE, MAE
    y1_val_exp_variance = []
    y1_val_mse = []
    y1_val_r2 = []

    y2_val_exp_variance = []
    y2_val_mse = []
    y2_val_r2 = []
    
    # Folds for cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # store the train & test index
    train_indices = []
    val_indices = []
    
    train_count = 0
    # main loop with cv-folding
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y)):

        # prepare data for training and validating in each fold
        print('Fold {}: num_train_ids={}, num_val_ids={}'.format(fold, len(train_ids), len(val_ids)))
        X_train, y_train, X_val, y_val = X[train_ids], y[train_ids], X[val_ids], y[val_ids]
        X_train_standard, X_val_standard = preprocess_standarization(X_train, X_val)
        
        # append the indices
        train_indices.append(train_ids)
        val_indices.append(val_ids)
        
        # fit the model
        model.fit(X_train_standard, y_train)

        # predict and evaluate performance on the val set
        y_val_pred = model.predict(X_val_standard)
        y1_exp_variance = explained_variance_score(y_val[:, 0], y_val_pred[:, 0])
        y1_r2 = r2_score(y_val[:, 0], y_val_pred[:, 0])
        y1_mse = mean_squared_error(y_val[:, 0], y_val_pred[:, 0])

        y2_exp_variance = explained_variance_score(y_val[:, 1], y_val_pred[:, 1])
        y2_r2 = r2_score(y_val[:, 1], y_val_pred[:, 1])
        y2_mse = mean_squared_error(y_val[:, 1], y_val_pred[:, 1])


        y1_val_exp_variance.append(y1_exp_variance)
        y1_val_r2.append(y1_r2)
        y1_val_mse.append(y1_mse)

        y2_val_exp_variance.append(y2_exp_variance)
        y2_val_r2.append(y2_r2)
        y2_val_mse.append(y2_mse)

        # print to check the errors
        print('\t Validation - Y1  {:2d}: ExpVar={:7.3f}, R2={:7.3f}, MSE={:7.3f}'.format(train_count, y1_val_exp_variance[train_count], y1_val_r2[train_count], y1_val_mse[train_count]))
        print('\t Validation - Y2  {:2d}: ExpVar={:7.3f}, R2={:7.3f}, MSE={:7.3f}'.format(train_count, y2_val_exp_variance[train_count], y2_val_r2[train_count], y2_val_mse[train_count]))

        train_count += 1
    
    print('--------------------------------------------------------')

    y1_val_exp_variance = np.array(y1_val_exp_variance)
    y1_val_r2 = np.array(y1_val_r2)
    y1_val_mse = np.array(y1_val_mse)

    y2_val_exp_variance = np.array(y2_val_exp_variance)
    y2_val_r2 = np.array(y2_val_r2)
    y2_val_mse = np.array(y2_val_mse)

    print()
    print("Performance on Validation set - Y1: ExpVar %.3f (+- %.3f)" % (y1_val_exp_variance.mean(),y1_val_exp_variance.std()))
    print("Performance on Validation set - Y1: R2     %.3f (+- %.3f)" % (y1_val_r2.mean(),y1_val_r2.std()))
    print("Performance on Validation set - Y1: MSE    %.3f (+- %.3f)" % (y1_val_mse.mean(),y1_val_mse.std()))

    print("Performance on Validation set - Y2: ExpVar %.3f (+- %.3f)" % (y2_val_exp_variance.mean(),y2_val_exp_variance.std()))
    print("Performance on Validation set - Y2: R2     %.3f (+- %.3f)" % (y2_val_r2.mean(),y2_val_r2.std()))
    print("Performance on Validation set - Y2: MSE    %.3f (+- %.3f)" % (y2_val_mse.mean(),y2_val_mse.std()))

    print()
    
    ret_model = model
    return ret_model

# Define retrain model on whole dataset
def retrain_model(model, X, y):

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    trained_model = model.fit(X, y)
    y_train_pred = trained_model.predict(X)

    # Evaluate performance
    y1_exp_variance = explained_variance_score(y[:, 0], y_train_pred[:, 0])
    y1_r2 = r2_score(y[:, 0], y_train_pred[:, 0])
    y1_mse = mean_squared_error(y[:, 0], y_train_pred[:, 0])

    y2_exp_variance = explained_variance_score(y[:, 1], y_train_pred[:, 1])
    y2_r2 = r2_score(y[:, 1], y_train_pred[:, 1])
    y2_mse = mean_squared_error(y[:, 1], y_train_pred[:, 1])

    print('\t Retrain on whole dataset - Y1: ExpVar={:7.3f}, R2={:7.3f}, MSE={:7.3f}'.format(y1_exp_variance, y1_r2, y1_mse))
    print('\t Retrain on whole dataset - Y2: ExpVar={:7.3f}, R2={:7.3f}, MSE={:7.3f}'.format(y2_exp_variance, y2_r2, y2_mse))

    return trained_model

# Define prediction on unknown datset
def predict_model(model, X_test):
    y_test_pred = model.predict(X_test)
    return y_test_pred

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                             solver='adam', max_iter=500, random_state=42)
trained_model = training_model(model, X, y)

# Retrain model on whole dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
retrained_model = retrain_model(model, X_train_scaled, y)