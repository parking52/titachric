
import pandas as pd
from clag.data_container import LassoModel



test_data = pd.read_csv('../data/header_test.csv')

def evaluate(model_name):
    model = model_name()
    model.train()
    predictions = model.predict(test_data[model.features])
    print(predictions)

 
def main():
    evaluate(LassoModel)


if __name__ == '__main__':
    main()