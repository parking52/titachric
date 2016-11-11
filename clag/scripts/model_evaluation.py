
import pandas as pd
from clag.models.linear_classifier_with_lasso import LassoModel
from clag.models.Mendie import Mendie


def evaluate(model):
    model_instance = model()
    model_instance.train()
    pass
    # predictions = model_instance.predict(model_instance.model, model_instance.model)
    # print(predictions)


def main():
    evaluate(LassoModel)
    evaluate(Mendie)

if __name__ == '__main__':
    main()