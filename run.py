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

        # y = W2 * ReLU * (W1 * X + b1) + b2
        print('---------------------------------')
        print('The MLP coefs for the fist (hidden) layer:')
        print(' + Coefficients: shape=', final_model.coefs_[0].shape)
        print(' + Intercept: shape=', final_model.intercepts_[0].shape)
        print('---------------------------------')
        print('The MLP coefs for the output layer:')
        print(' + Coefficients: shape=', final_model.coefs_[1].shape)
        print(' + Intercept: shape=', final_model.intercepts_[1].shape)
        print('---------------------------------')

        print('Thong tin mo hinh MLP:')
        print("  + Num. layers:", model.n_layers_)
        print("  + Num. outputs:", model.n_outputs_)
        print("  + Hidden layer sizes:", model.hidden_layer_sizes)
        print("  + Activation function:", model.activation)
        print(" ")

        for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
            print(f"Layer {i+1}: ")
            print(f"  + Weights shape: {w.shape}")
            print(f"  + Biases shape: {b.shape}")
        print('---------------------------------')

        # Ghi ra file cac ma tran phi tuyen cua model MLP
        W1 = final_model.coefs_[0]
        b1 = final_model.intercepts_[0]
        W2 = final_model.coefs_[1]
        b2 = final_model.intercepts_[1]

        data_heso = {}
        for i in range(W1.shape[0]):
            data_heso[f'W1_row{i}'] = W1[i]

        for i in range(W2.shape[0]):
            data_heso[f'W2_row{i}'] = W2[i]
        
        data_heso['b1_col0'] = b1
        data_heso['b2_col0'] = b2

        df_heso = pd.DataFrame(data_heso)
        print('Ghi cac ma tran he so ra file: heso_mlp.csv')
        df_heso.to_csv('./heso_mlp.csv', index=False)
        print('---------------------------------')

    