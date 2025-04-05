import optuna
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Preprocessing the dataset
def preprocess_standarization(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

# set seeds for reproducibility
def set_seeds(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)


# Define Optuna objective function
def objective(trial):

    # Hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_sizes = [trial.suggest_int(f"n_units_l{i}", 8, 64) for i in range(n_layers)]

    learning_rate_init = trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-1)

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
        
        set_seeds()

        # append the indices
        train_indices.append(train_ids)
        val_indices.append(val_ids)
        
        # Build model
        model = MLPRegressor(
            hidden_layer_sizes=tuple(layer_sizes),
            activation='relu',
            learning_rate_init=learning_rate_init,
            solver='adam',
            max_iter=300,
            random_state=42
        )

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
    
    y1_val_exp_variance = np.array(y1_val_exp_variance)
    y1_val_r2 = np.array(y1_val_r2)
    y1_val_mse = np.array(y1_val_mse)

    y2_val_exp_variance = np.array(y2_val_exp_variance)
    y2_val_r2 = np.array(y2_val_r2)
    y2_val_mse = np.array(y2_val_mse)

    current_val_exp_variance = (float(np.mean(y1_val_exp_variance)) + float(np.mean(y2_val_exp_variance)))/2
    current_val_mse = (float(np.mean(y1_val_mse)) +float(np.mean(y2_val_mse)))/2

    return current_val_mse

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

    print('\t Retrain on whole dataset - Y1: ExpVar={:7.4f}, R2={:7.4f}, MSE={:7.4f}'.format(y1_exp_variance, y1_r2, y1_mse))
    print('\t Retrain on whole dataset - Y2: ExpVar={:7.4f}, R2={:7.4f}, MSE={:7.4f}'.format(y2_exp_variance, y2_r2, y2_mse))

    return trained_model, y_train_pred

# set_seeds()

# Run Optuna optimization
# study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
# study.optimize(objective, n_trials=30)

# set_seeds()

# Print best parameters
# print(" Best params: {}".format(study.best_params))

# Rebuild best model using best params
# best_params = study.best_params
# best_layer_sizes = tuple([best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])])

# best_model = MLPRegressor(
#     hidden_layer_sizes=best_layer_sizes,
#     learning_rate_init=best_params["learning_rate_init"],
#     solver='adam',
#     max_iter=300,
#     random_state=42
# )
###  Best params: {'n_layers': 3, 'n_units_l0': 54, 'n_units_l1': 47, 'n_units_l2': 62, 'learning_rate_init': 0.025153863501814786}

# Retrain using your existing function
# trained_model = training_model(best_model, X, y)

# Final retraining on all data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# retrained_model, y_train_pred = retrain_model(best_model, X_scaled, y)

# df_y_train_pred =pd.DataFrame(y_train_pred) 

# df_y_train_pred.to_csv("result_MLP.csv", index = False, header = ['Y1', 'Y2'])
