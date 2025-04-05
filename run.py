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
                        default='LinearRegression',
                        help="Name of regression models, e.g., Linear, DecisionTree, RandomForest, GB, SVM, MLP")
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

    if model_name == 'LinearRegression':

        print('---------------------------------')
        print('Run Linear Regression model')
        print('---------------------------------')
        model = LinearRegression()
        trained_model = training_model(model, X, y)


    elif model_name == 'DecisionTree':
        print('Run Decision Decision Tree model')

        # Training model
        model = DecisionTreeRegressor()
        trained_model = training_model(model, X, y)

        # Retrain model on whole dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        retrained_model, y_train_pred = retrain_model(trained_model, X_train_scaled, y)

        #Write the results to csv file
        df_y_train_pred =pd.DataFrame(y_train_pred)
        df_y_train_pred.to_csv("result_DecisionTree.csv", index = False, header = ['Y1', 'Y2']) 

    elif model_name == 'RandomForest':
        print('Run Random Forest model')

        # Training model
        model = RandomForestRegressor()
        trained_model = training_model(model, X, y)

        # Retrain model on whole dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        retrained_model, y_train_pred = retrain_model(trained_model, X_train_scaled, y)

        #Write the results to csv file
        df_y_train_pred =pd.DataFrame(y_train_pred)
        df_y_train_pred.to_csv("result_RandomForest.csv", index = False, header = ['Y1', 'Y2'])            

    elif model_name == 'SVM':
        print('Run SVM model')

    
    elif model_name == 'MLP':
        print('Run MLP model')
        # model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
        #                      solver='adam', max_iter=500, random_state=42)
        # trained_model = training_model(model, X, y)

        # # Retrain model on whole dataset
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X)
        # retrained_model = retrain_model(model, X_train_scaled, y)