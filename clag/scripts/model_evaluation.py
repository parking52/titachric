
import pandas as pd
from clag.models.linear_classifier_with_lasso import LassoModel


def evaluate(model):
    model_instance = model()
    model_instance.train()
    pass
    # predictions = model_instance.predict(model_instance.model, model_instance.model)
    # print(predictions)


def main():
    evaluate(LassoModel)


if __name__ == '__main__':
    main()