import numpy as np
import pandas as pd
from time import perf_counter
#from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import copy

from xpc.utils import series
from xpc.utils import plots
from xpc.utils import models
from xpc.utils import data
from xpc.shapley import gams
from xpc.shapley import explainers
from xpc.shapley import analysis



def get_background_data(model, background, background_contribs=None, background_y=None, col_mapping=None, **kwargs):
    """returns dictionary of background information"""
    
    background_dict = {"background_y":background_y, "background_contribs":background_contribs}
    
    #true load
    if kwargs.get("background_true_expectation") is None:
        if background_y is not None:
            background_true_expectation = np.mean(background_y.values)
            background_dict["background_true_expectation"] = background_true_expectation
    else:
        background_dict["background_true_expectation"] = kwargs["background_true_expectation"]
            
    #true contributions
    if kwargs.get("background_true_mean_percents") is None:
        if background_contribs is not None: 
            background_true_mean_percents = np.array([np.mean(background_contribs[key])/background_true_expectation for key in background_contribs])
            background_dict["background_true_mean_percents"] = background_true_mean_percents
    else:
        background_dict["background_true_mean_percents"] = kwargs["background_true_mean_percents"]
    
    #gam contributions
    if model.model_class == "gam":
        if kwargs.get("background_spline_contribs") is None:
            #spline predictions
            background_predictions_all, background_predictions, background_expectation, background_spline_contribs, background_spline_means = gams.get_spline_contributions(background, model.predict)
            #aggregating
            merged_background_spline_contribs = series.aggregate_series(background_spline_contribs, col_mapping)
            #heightening
            background_spline_deltas, background_spline_mins, background_spline_min_sum = series.get_deltas(merged_background_spline_contribs, do_all=kwargs.get("do_all",True), alpha=kwargs.get("alpha",1))
            #percents
            background_spline_mean_percents = np.array([np.mean(background_spline_deltas[key])/(background_expectation-background_spline_min_sum) for key in background_spline_deltas])

            background_dict = {**background_dict, "background_predictions":background_predictions, "background_expectation":background_expectation, "background_predictions_all":background_predictions_all, "background_spline_contribs":background_spline_contribs, "background_spline_means":background_spline_means, "background_spline_mins":background_spline_mins,"background_spline_mean_percents":background_spline_mean_percents,"merged_background_spline_contribs":merged_background_spline_contribs}
        else:
            background_dict = {**background_dict, "background_predictions":kwargs["background_predictions"], "background_expectation":kwargs["background_expectation"], "background_predictions_all":kwargs["background_predictions_all"], "background_spline_contribs":kwargs["background_spline_contribs"], "background_spline_means":kwargs["background_spline_means"], "background_spline_mins":kwargs["background_spline_mins"],"background_spline_mean_percents":kwargs["background_spline_mean_percents"],"merged_background_spline_contribs":kwargs["merged_background_spline_contribs"]}
    
    #predictions
    if background_dict.get("background_predictions") is None:
        if kwargs.get("background_predictions") is None:
            background_predictions = model.predict(background)
            background_expectation = np.mean(background_predictions)
            background_dict["background_predictions"] = background_predictions
            background_dict["background_expectation"] = background_expectation
        else:
            background_dict["background_predictions"] = kwargs["background_predictions"]
            background_dict["background_expectation"] = kwargs["background_expectation"]
    return background_dict



def get_X_data(model, X, background, y=None, true_contribs=None, background_dict={}, col_mapping=None, do_exact=False, do_plot=False, **kwargs):
    """returns dictionary of X information"""
    
    assert(hasattr(model,"MyModelMarker"))
    X_dict = {"y":y, "true_contribs":true_contribs}
    
    #true contributions
    if kwargs.get("true_percents") is None:
        if true_contribs is not None:
            true_percents = series.get_percents(true_contribs)
            X_dict["true_percents"] = true_percents
    else:
        X_dict["true_percents"] = kwargs["true_percents"]
    
    #gam contributions
    if model.model_class == "gam":
        if kwargs.get("spline_contribs") is None:
            predictions_all, predictions, expectation, spline_contribs, spline_means = gams.get_spline_contributions(X, model.predict)
            merged_spline_contribs = series.aggregate_series(spline_contribs, col_mapping)
            spline_delta_df, spline_df_mins, spline_min_sum = series.get_deltas(merged_spline_contribs, background_dict.get("background_spline_mins"), kwargs.get("do_all",True), kwargs.get("alpha",1))
            spline_percents = series.get_percents(spline_delta_df)

            X_dict = {**X_dict, "predictions":predictions, "predictions_all":predictions_all, "spline_contribs":spline_contribs, "spline_delta_df":spline_delta_df,"spline_df_mins":spline_df_mins,"spline_min_sum":spline_min_sum,"spline_percents":spline_percents}
        else:
            X_dict = {**X_dict, "predictions":kwargs.get("predictions"), "predictions_all":kwargs["predictions_all"], "spline_contribs":kwargs["spline_contribs"], "spline_delta_df":kwargs["spline_delta_df"],"spline_df_mins":kwargs["spline_df_mins"],"spline_min_sum":kwargs["spline_min_sum"],"spline_percents":kwargs["spline_percents"]}
    
    #predictions
    if X_dict.get("predictions") is None:
        if kwargs.get("predictions") is None:
            predictions = model.predict(X)
            X_dict["predictions"] = predictions
        else:
            X_dict["predictions"] = kwargs["predictions"]
    
    #exact shap contributions
    if kwargs.get("exact_percents") is not None:
        X_dict["exact_percents"] = kwargs["exact_percents"]
    elif do_exact:
        explainer = explainers.Shapley(model, algo="exact", col_mapping=col_mapping, **kwargs)
        explainer.fit(background, background_dict, **kwargs)
        exact_parts = explainer(X, X_dict, **kwargs)
        exact_percents = explainer.percents
        X_dict["exact_percents"] = exact_percents
    
    #plots
    if do_plot:
        #contributions
        if y is not None:
            plots.show_predictions(y, X_dict["predictions"], title="Load and predictions")
        if true_contribs is not None:
            plots.show_sum(true_contribs, title="SMACH contributions")
            plots.show_sum(X_dict["true_percents"],title="SMACH %")
        if model.model_class == "gam":
            plots.show_sum(X_dict["spline_delta_df"], offset=X_dict["spline_min_sum"], title="GAM contributions")
            plots.show_sum(X_dict["spline_percents"],title="GAM %")
            
        #errors
        key = next(iter(true_percents.keys()))
        print("==========ERRORS==========")
        if y is not None:
            error = series.score(X_dict["predictions"], y, relative=True, percents=True)
            error_abs = series.score(X_dict["predictions"], y, relative=True, percents=True, absolute=True)
            print(f"Mean model error: {error:.2f} %")
            print(f"Mean model absolute error: {error_abs:.2f} %")
        if model.model_class == "gam" and true_contribs is not None:
            print("")
            error = series.score(X_dict["spline_percents"][key], X_dict["true_percents"][key], percents=True)
            error_abs = series.score(X_dict["spline_percents"][key], X_dict["true_percents"][key], percents=True, absolute=True)
            print(f"Mean error (GAM to SMACH): {error:.2f} %")
            print(f"Mean absolute error (GAM to SMACH): {error_abs:.2f} %")
        if do_exact and true_contribs is not None:
            print("")
            error = series.score(X_dict["exact_percents"][key], X_dict["true_percents"][key], percents=True)
            error_abs = series.score(X_dict["exact_percents"][key], X_dict["true_percents"][key], percents=True, absolute=True)
            print(f"Mean error (SHAP to SMACH): {error:.2f} %")
            print(f"Mean absolute error (SHAP to SMACH): {error_abs:.2f} %")
    
    return X_dict