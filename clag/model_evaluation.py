
import pandas as pd
from clag.model_container import LassoModel



test_data = pd.read_csv('/home/chris/python_projects/titachric/data/header_test.csv')
train_data = pd.read_csv('/home/chris/python_projects/titachric/data/header_train.csv')


def evaluate(model_name):
    model_instance = model_name()
    model_instance.train(train_data)
    predictions = model_instance.predict(model_instance.model, train_data)
    print(predictions)


def main():
    evaluate(LassoModel)


if __name__ == '__main__':
    main()