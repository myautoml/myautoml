from sklearn.model_selection import train_test_split


def load_training_data():
    raise NotImplementedError("Loading of training data not implemented")
    # return x_train, y_train


def load_prediction_data():
    raise NotImplementedError("Loading of prediction data not implemented")
    # return x_test, y_test


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test
