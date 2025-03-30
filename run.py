import argparse
import numpy as np

from model_scikit import *

exit()

if __name__ == '__main__':
    """
    Run the main.py file to start the program:
        + Process the input arguments
        + Read data
        + Preprocess data
        + Train models
        + Prediction
    """

    # ----------------------------------------------------
    # Process the arguments
    # ----------------------------------------------------
    parser = argparse.ArgumentParser()

    
    parser.add_argument("-llm", "--model_name", type=str,
                        default='MLP',
                        help="Name of LL model for training the phenotype prediction: Linear, GB, SVM, MLP")
    

    args = vars(parser.parse_args())

    # ----------------------------------------------------
    # Parsing the input arguments
    # ----------------------------------------------------

    model_name = args["model_name"]

    # print('-----------------------------------------------')
    # print('Input arguments: ')
    # print('   + model: {}'.format(model))


    if model_name == 'Linear':
        print('Run Linear Regression model')

        # Training model
        model = LinearRegression()
        trained_model = training_model(model, X, y)

        # Retrain model on whole dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        retrained_model = retrain_model(model, X_train_scaled, y)

        # Prediction on unknow data
        # X_test_scaled = scaler.transform(X_test)
                   

    elif model_name == 'SVM':
        print('Run SVM model')

    
    elif model_name == 'MLP':
        print('Run MLP model')
        model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                             solver='adam', max_iter=500, random_state=42)
        trained_model = training_model(model, X, y)

        # Retrain model on whole dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        retrained_model = retrain_model(model, X_train_scaled, y)