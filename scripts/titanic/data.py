from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_data():
    x, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    y = y.cat.codes.astype('bool')
    return x, y


def load_training_data():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=123)
    return x_train, y_train


def load_prediction_data():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=123)
    return x_test, y_test
