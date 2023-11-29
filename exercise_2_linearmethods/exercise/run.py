import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mse(y_prediction: np.ndarray, y: np.ndarray):
    """Returns the mean squared error for the predictions y_prediction and the real values y.
    y_prediction and y is of shape (N, 1)."""
    # TODO
    mean_square = (y_prediction - y)**2
    mse = mean_square.mean()
    return mse
    raise NotImplementedError()


def gradient_descent(X: np.ndarray, y: np.ndarray, iterations = 1000, learning_rate = 0.1, stopping_threshold = 1e-6):
    """Returns the optimal bias b and weight       
    ((( Zur Berechnung wird wie folgt vorgegangen
1. Starte an einem zufälligen Punkt.
2. Berechne die Vorhersage der Datenpunkte mit
den aktuellen Modellgewichten.
3. Gehe zum nächsten Punkt in Abhängigkeit der
Learning Rate (Aktualisierung der
Modellgewichte).
4. Prüfe, ob das Minimum erreicht wurde.
5. Wenn nein, gehe zu Schritt 2. Wenn ja, fertig. )))
    for given data set X of shape (N, 2) and target values y of shape (N, 1)"""
    # For the following implementation, we need the feature values and target values as arrays, each of shape (N)
    x = X[:,1]
    y = y[:,0]

    # STEP X: Initializing (random) weight, (random) bias, learning rate and iterations
    current_weight = 8 # TODO: choose different/random value, if you want
    current_bias = 3 # TODO: choose different/random value, if you want
    iterations = iterations
    learning_rate = learning_rate

    n = float(len(x)) # Cache n so that it is not recalculated over and over again
     
    costs = []
    weights = []
    previous_cost = None

    # Estimation of optimal parameters
    for i in range(iterations):
         
        ### STEP X: Updating the weights and bias values (calculating the next point)

        # Calculating predictions
        y_prediction = current_weight * x+ current_bias # TODO: Add formula here
        # Calculating the current cost
        current_cost = mse(y_prediction, y)
 
        costs.append(current_cost)
        weights.append(current_weight)

        # Calculating the derivatives
        weight_derivative = None # TODO: Add formula here
        bias_derivative = None # TODO: Add formula here
        
        # ax + b a ist gewichtunv 
        # Calculating the new values
        current_weight = None # TODO: Add formula here
        current_bias = None # TODO: Add formula here
                 
        # Printing the parameters for each 1000th iteration, uncomment if needed
        # print(f"Iteration {i+1}: Cost {current_cost}, Weight {current_weight}, Bias {current_bias}")

        #### STEP X: check if cost got still reduced
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break ### STEP X: break if minimum reached
        previous_cost = current_cost
     
    # Visualizing the weights and cost at for all iterations
    fig = plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    fig.show()

    y_pred = None # TODO: Add formula here, hint: same as in line 37

    # Plotting the regression line
    fig2 = plt.figure(figsize = (8,6))
    plt.scatter(x, y, marker='o', color='red')
    plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], linestyle='dashed')
    plt.xlabel("X (scaled)")
    plt.ylabel("Y")
    fig2.show()
    plt.show()
     
    return current_weight, current_bias

def closed_form_solution(X: np.ndarray, y: np.ndarray):
    """Returns the optimal model parameters theta for the feature matrix X and the real values y.
    X is of shape (N, M) where N denotes the number of data points and M the number of features (+1 for bias).
    y is of shape (N, 1)."""
    raise NotImplementedError()


def question_1():
    """Returns the correct answer to question 1."""
    # Gucken Sie sich die Scatter-Plots der features [2, 3, 4, 5, 9] an
    # Welches Feature lässt sich am besten mit einem linearen Modell darstellen
    # Antwort aus INDUS CHAS NOX RM TAX
    # TODO
    return ""


def question_2():
    """Returns the correct answer to question 2."""
    # Welches Feature lässt sich am schlechtesten mit einem linearen Modell darstellen
    # Antwort aus INDUS CHAS NOX RM TAX
    # TODO
    return ""


def question_3():
    """Returns the correct answer to question 3."""
    # Gucken sie sich das Modell für alle Features an
    # Welches Feature hat den größten Einfluss
    # Antwort aus INDUS CHAS NOX RM TAX
    # TODO
    return ""


def question_4():
    """Returns the correct answer to question 4."""
    # Welches Feature hat den geringsten Einfluss
    # Antwort aus INDUS CHAS NOX RM TAX
    # TODO
    return ""


def min_max_scaling(X):
    """Applies min-max-scaling to the matrix X.
    Every column is scaled to the interval (0, 1)."""
    return (X - X.min(0)) / (X.max(0) - X.min(0))


def get_linear_regression_training_set_from_df(df: pd.DataFrame, columns: list):
    """Returns the training set X and the target column y from the data frame df.
    columns is a list of integers indicating which features should be included in X.
    y is the target column (must be the last column of the data frame).
    The first column of X is filled with 1.
    The shape of X is (N, M) where N denotes the number of data points
    and M the number of features + 1 (added bias term)."""
    # extract feature columns from data
    X = df.iloc[:, columns].to_numpy()
    # apply scaling to data set so we can better compare different features
    X = min_max_scaling(X)
    # For linear regression we need to add an additional feature which is always 1 for the bias term
    # np.c_ concatenates two 2d arrays along the second axis, which can be seen as adding additional columns
    # np.ones(shape) creates an array of shape shape filled with 1
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # extract target column from data; note that y needs to be of shape (n, 1), i.e. a matrix with a single column
    # this is called a column vector, while a matrix of shape (1, n) is called a row vector
    y = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    return X, y


def plot_features_from_df(df: pd.DataFrame, columns: list, n_plot_columns: int = 2):
    """Plots the features of df indicated by columns.
    columns is a list of integers indicating which features should be plotted.
    y-Axis will always be the target column (must be the last column of the data frame)"""
    n_ax_columns = n_plot_columns
    n_ax_rows = int(len(columns) / n_ax_columns)
    n_ax_rows += 1 if len(columns) % n_ax_columns > 0 else 0
    fig, ax = plt.subplots(n_ax_rows, n_ax_columns, figsize=(3 * n_ax_columns, 3 * n_ax_rows), sharey=True)
    for column, ax_ in zip(columns, ax.flat):
        ax_.scatter(df.iloc[:, column], df.iloc[:, -1])
        ax_.set_title(f"{df.columns[column]} ({column})")
    if len(columns) % n_ax_columns > 0:
        for i in range(n_ax_columns - len(columns) % n_ax_columns):
            ax.flat[-(i + 1)].set_axis_off()
    fig.tight_layout()
    # fig.canvas.manager.set_window_title('Features')
    plt.show()


def get_predictions(theta, X):
    """Returns the predictions of the linear model theta for the data set X.
    theta is of shape (M, 1), where M denotes the number of model parameters (i.e. number of features + 1 [bias]).
    X is of shape (N, M)."""
    return X.dot(theta)


if __name__ == "__main__":
    # import data set
    import os

    print(os.getcwd())
    df = pd.read_csv("../data/boston_house_prices.csv")
    features = [2, 3, 4, 5, 9]
    
    # plot features
    plot_features_from_df(df, features, 2)
    
    # MSE of single features
    print("MSE single features:")
    for i in features:
        X, y = get_linear_regression_training_set_from_df(df, [i])
        theta = closed_form_solution(X, y)
        print(f"{df.columns[i]}: {mse(get_predictions(theta, X), y): 0.2f}")
    print("\n")
    X, y = get_linear_regression_training_set_from_df(df, features)
    theta = closed_form_solution(X, y)
    print("Model:")
    print(" +\n\t".join(
        [f"{np.round(coef, 2)} * ({feature_name})"
         for feature_name, coef in zip(["Bias"] + list(df.columns[features]), theta.flatten())]
    ))
    print(f"MSE Model: {mse(get_predictions(theta, X), y): 0.2f}")
