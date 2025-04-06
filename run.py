import argparse
import numpy as np

from src.model_scikit import *
from src.model_mlp import *

if __name__ == '__main__':

    # ----------------------------------------------------
    # Process the arguments
    # ----------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        default='MLP',
                        help="Name of regression models, e.g., LR, DT, RF, SVM, MLP")
    args = vars(parser.parse_args())

    # ----------------------------------------------------
    # Parsing the input arguments
    # ----------------------------------------------------
    model_name = args["model"]

    # ----------------------------------------------------
    # Load and preprocess dataset
    # ----------------------------------------------------
    X = pd.read_csv("./data/X.csv").to_numpy()
    y = pd.read_csv("./data/y.csv").to_numpy()
    X_test = pd.read_csv("./data/X_test.csv").to_numpy()
    y_test = pd.read_csv("./data/y_test.csv").to_numpy()

    if model_name == 'LR':

        print('---------------------------------')
        print('Run Linear Regression model')
        print('---------------------------------')
        model = LinearRegression()
        trained_model = training_model(model, X, y)

        # Retrain model on whole dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_model, y_final_pred = retrain_model(trained_model, X_scaled, y)
        # Test the model
        y_pred = predict_model(final_model, X_test, y_test)

        print('---------------------------------')
        print('The regression function:')
        print(' + Coefficients: ', final_model.coef_)
        print(' + Intercept: ', final_model.intercept_)
        print('---------------------------------')
        print(' + y1 = {:.4f} * x1 + {:.4f} * x2 + {:.4f} * x3 + {:.4f} * x4 + {:.4f} * x5 + {:.4f}'.format(
            final_model.coef_[0][0], final_model.coef_[0][1], final_model.coef_[0][2], final_model.coef_[0][3], final_model.coef_[0][4], final_model.intercept_[0]))
        print(' + y2 = {:.4f} * x1 + {:.4f} * x2 + {:.4f} * x3 + {:.4f} * x4 + {:.4f} * x5 + {:.4f}'.format(
            final_model.coef_[1][0], final_model.coef_[1][1], final_model.coef_[1][2], final_model.coef_[1][3], final_model.coef_[1][4], final_model.intercept_[1]))
        print('---------------------------------')

    elif model_name == 'DT':

        print('---------------------------------')
        print('Run Decision Tree model')
        print('---------------------------------')
        model = DecisionTreeRegressor()
        trained_model = training_model(model, X, y)

        # Retrain model on whole dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_model, y_final_pred = retrain_model(trained_model, X_scaled, y)
        # Test the model
        y_pred = predict_model(final_model, X_test, y_test)


    elif model_name == 'RF':
        
        print('---------------------------------')
        print('Run Random Forest model')
        print('---------------------------------')
        model = RandomForestRegressor()
        trained_model = training_model(model, X, y)

        # Retrain model on whole dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_model, y_final_pred = retrain_model(trained_model, X_scaled, y)
        # Test the model
        y_pred = predict_model(final_model, X_test, y_test)       

    elif model_name == 'SVM':
    
        print('---------------------------------')
        print('Run Support Vector Machine model')
        print('---------------------------------')

    
    elif model_name == 'MLP':
        
        print('---------------------------------')
        print('Run MLP model')
        print('---------------------------------')

        model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                             solver='adam', max_iter=500, random_state=42)
        trained_model = training_model(model, X, y)

        # # Retrain model on whole dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_model, y_final_pred = retrain_mlp_model(trained_model, X_scaled, y)

        # Test the model
        y_pred = predict_mlp_model(final_model, X_test, y_test)

        print('---------------------------------')
        print('The MLP coefs for predicting Y1:')
        print(' + Coefficients: ', final_model.coefs_[0])
        print(' + Intercept: ', final_model.intercepts_[0])
        print('---------------------------------')
        print('The MLP coefs for predicting Y2:')
        print(' + Coefficients: ', final_model.coefs_[1])
        print(' + Intercept: ', final_model.intercepts_[1])
        print('---------------------------------')
        