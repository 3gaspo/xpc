import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

import opera
from xpc.utils import models
from xpc.shapley import explainers

def get_mixture(models_dict, X, y, loss_gradient=True):
    """computes mixture weights based on models' losses on (X,y)"""
    expert_predictions = {}
    for model_name, model in models_dict.items():
        expert_predictions[model_name] = model.predict(X)
    expert_predictions = pd.DataFrame(expert_predictions)
    mixture = opera.Mixture(y=y, experts=expert_predictions, loss_type="mse", loss_gradient=loss_gradient)
    return mixture


def get_mixture_predictions(models_dict, mixture, X):
    """computes prediction using mixture of models"""
    expert_predictions = {}
    #expert_names = list(mixture.experts_names)
    for model_name, model in models_dict.items():
        expert_predictions[model_name] = model.predict(X)
    predictions = mixture.predict(pd.DataFrame(expert_predictions))
    return predictions.reshape(-1)


def get_mixture_model(models_dict, X, y, loss_gradient=True, do_print=False):
    """returns model as a mixture of models, wrapped as MyModel class"""
    t1 = perf_counter()
    mixture = get_mixture(models_dict, X, y, loss_gradient=True)
    t2 = perf_counter()
    if do_print:
        print(f"Mixture done in {(t2-t1)/60:.3f} min")
    mixture_model = models.MyModel(lambda x: get_mixture_predictions(models_dict,mixture,x))
    mixture_model.mixture = mixture
    return mixture_model


def plot_mixture(mixture_model):
    """returns mixture plot from opera package"""
    mixture_model.mixture.plot_mixture()

##shap
def get_mixture_shap_values(models_dict, mixture, X, background, algo=None, col_mapping=None, background_dict={}, X_dict={}, **kwargs):
    """returns Shapley values of mixture."""
    
    if col_mapping is None:
        shap_cols = background.columns
    else:
        shap_cols = list(col_mapping.keys())
        
    parts_mix = {col: np.array([0. for k in range(X.shape[0])]) for col in shap_cols}
    weights = mixture.w
    
    for model_name, model in models_dict.items():
        explainer = explainers.Shapley(model, algo=algo, col_mapping=col_mapping, **kwargs)
        explainer.fit(background, background_dict, **kwargs)
        parts = explainer(X, X_dict, **kwargs)

        for part_name in parts:
            parts_mix[part_name] += parts[part_name] * weights[k]

    return pd.DataFrame(parts_mix,columns=shap_cols)