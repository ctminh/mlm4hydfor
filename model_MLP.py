import optuna
from sklearn.neural_network import MLPRegressor

from model_scikit import *

# Define Optuna objective function
def objective(trial):

    # Hyperparameters
    # n_layers = trial.suggest_int("n_layers", 1, 3)
    # layer_sizes = [trial.suggest_int(f"n_units_l{i}", 64, 256) for i in range(n_layers)]
    outfactor = trial.suggest_float('outfactor', 0.05, 0.7, step=0.01)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    input_size = X.shape[1]
    layer_sizes = []

    for i in range(n_layers):
        next_size = max(2, int(input_size * outfactor))
        layer_sizes.append(next_size)
    
    learning_rate_init = trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-1)

    # Build model
    model = MLPRegressor(
        hidden_layer_sizes=tuple(layer_sizes),
        activation='relu',
        learning_rate_init=learning_rate_init,
        solver='adam',
        max_iter=300,
        random_state=42
    )

    # Train model using your existing CV-based function
    trained = training_model(model, X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_pred = trained.predict(X_scaled)

    # Compute average metrics
    y1_r2 = r2_score(y[:, 0], y_pred[:, 0])
    y2_r2 = r2_score(y[:, 1], y_pred[:, 1])

    y1_exp_variance = explained_variance_score(y[:, 0], y_pred[:, 0])
    y2_exp_variance = explained_variance_score(y[:, 1], y_pred[:, 2])

    # return (y1_r2 + y2_r2) / 2
    return (y1_exp_variance + y2_exp_variance) / 2
    # return y2_exp_variance

# Run Optuna optimization
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30)

# Print best parameters
print(" Best params: {}".format(study.best_params))

# Rebuild best model using best params
best_params = study.best_params
best_layer_sizes = tuple([best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])])

best_model = MLPRegressor(
    hidden_layer_sizes=best_layer_sizes,
    learning_rate_init=best_params["learning_rate_init"],
    solver='adam',
    max_iter=300,
    random_state=42
)

# Retrain using your existing function
trained_model = training_model(best_model, X, y)

# Final retraining on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
retrained_model = retrain_model(best_model, X_scaled, y)
