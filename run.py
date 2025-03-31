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
                        default='RandomForest',
                        help="Name of LL model for training the phenotype prediction: Linear, DecisionTree, RandomForest, GB, SVM, MLP")
    

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
        retrained_model, y_train_pred = retrain_model(trained_model, X_train_scaled, y)

        #Write the results to csv file
        df_y_train_pred =pd.DataFrame(y_train_pred)
        df_y_train_pred.to_csv("result_LinearRegression.csv", index = False, header = ['Y1', 'Y2'])

        # Prediction on unknow data
        # X_test_scaled = scaler.transform(X_test)

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