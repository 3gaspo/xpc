import numpy as np
import pandas as pd

from xpc.shapley import monte_carlo
from xpc.utils import data
from xpc.shapley import conditional



def get_spline_contributions(df, predict_fct):
    """returns spline predictions from gam model"""
    preds = predict_fct(df, return_all=True)
    predictions_all = preds.drop(columns=["EstimatedLoad"])
    predictions = preds["EstimatedLoad"] 
        
    expectation = np.mean(predictions)
    spline_contribs = {}
    spline_means = {}
    
    for col in predictions_all.columns:
        spline_contribs[col] = predictions_all[col]
        spline_means[col] = np.mean(spline_contribs[col])        
        
    return predictions_all, predictions, expectation, pd.DataFrame(spline_contribs), spline_means



def get_approximate_conditional(predict_fct, X, background, n_coalitions=10, n_approx=1, n_intervals=10, do_add_delta=False, intervals=None, features_map=None, **kwargs):
    """returns approximation of E(f(xj)|S) (- sigma(S) if do_add_delta=True) for S in sampled coalitions. Only relevant for monovariate GAM."""

    feature_names = list(background.columns)
    N, p = X.shape
    spline_values = {feature:[] for feature in feature_names}
    coalition_values = [[[] for i in range(N)] for j in range(p)]

    #sample coalitions
    sampled_coalitions = [[] for i in range(N)]
    do_sample_coalitions = (n_coalitions != 0)
    for i,x in enumerate(X.values):
        for j, feature in enumerate(feature_names):
            sampled_coalitions[i].append(monte_carlo.sample_coalitions(p, j, n_coalitions, monte_carlo=do_sample_coalitions))

    #predictions dataset
    prediction_dataset = []
    for i, x in enumerate(X.values):
        for j, feature in enumerate(feature_names):
            coalitions = sampled_coalitions[i][j]
            for k, coalition in enumerate(coalitions):
                all_neighbors = conditional.get_neighbors(x, background, coalition, intervals, features_map)
                if n_approx != 0:
                    neighbors = all_neighbors.iloc[np.random.choice(all_neighbors.shape[0], n_approx, replace=True)]
                else:
                    neighbors = all_neighbors
                prediction_dataset += list(neighbors.values)
                
                if do_add_delta:
                    neighbors_cup_j = conditional.get_neighbors(x, all_neighbors, [j], intervals, features_map)
                    if n_approx != 0:
                        neighbors_cup_j = neighbors_cup_j.iloc[np.random.choice(neighbors_cup_j.shape[0], n_approx, replace=True)]
                    prediction_dataset += list(neighbors_cup_j.values)
    
    prediction_dataset = pd.DataFrame(np.array(prediction_dataset),columns=X.columns)
    #preds
    neighbors_preds = predict_fct(prediction_dataset,return_all=True)

    #sum pi_S E(f(xj)|S) (- sigma(S) if do_add_delta=True) for sampled S
    pos = 0
    for i,x in enumerate(X.values):
        for j, feature in enumerate(feature_names):
            coalitions = sampled_coalitions[i][j]
            coalition_values = []
            for k,coalition in enumerate(coalitions):
                coalition_value = np.mean(neighbors_preds[feature][pos:pos+n_approx])
                pos += n_approx
                
                if do_add_delta:
                    remaining_k = [ind for ind in range(p) if (ind != j and ind not in coalition)]
                    for ktilde in remaining_k:
                        coalition_value += np.mean(neighbors_preds[feature_names[ktilde]][pos-n_approx:pos]) #E(fk|S)
                        coalition_value += -np.mean(neighbors_preds[feature_names[ktilde]][pos:pos+n_approx]) #E(fk|S+j)
                    pos += n_approx
                
                #pi(S)
                weight = monte_carlo.get_coalition_weight(coalition, p, monte_carlo=do_sample_coalitions)
                coalition_values.append(weight * coalition_value)
            if do_sample_coalitions:
                spline_values[feature].append(np.mean(coalition_values))
            else:
                spline_values[feature].append(np.sum(coalition_values))

        
    return pd.DataFrame(spline_values)