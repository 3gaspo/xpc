import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import itertools
from scipy.special import binom

from xpc.shapley import conditional


##background sampling

def sample_background(background, coalitions, n_approx, coalitions_id=None):
    """samples dataset of given size from background data"""
    if coalitions_id is None:
        sampled_background = []
        for coalition in coalitions:
            if n_approx == 0:
                sampled_background.append(background)
            else:
                sampled_background.append(background.iloc[np.random.choice(background.shape[0], n_approx, replace=True)].values)
        return np.array(sampled_background)
    else:
        sampled_background = {}
        for index, coalition in zip(coalitions_id, coalitions):
            if n_approx == 0:
                sampled_background[str(coalition)+str(index)] = background.copy
            else:
                sampled_background[str(coalition)+str(index)] = background.iloc[np.random.choice(background.shape[0], n_approx, replace=True)].values
        return sampled_background


def sample_conditional_background(background, x, coalitions, n_approx, intervals, features_map, coalitions_id=None):
    """samples dataset of given size from background data"""
    if coalitions_id is None:
        sampled_background = []
        for coalition in coalitions:
            neighbors = conditional.get_neighbors(x, background, coalition, intervals, features_map)
            sampled_background.append(neighbors.iloc[np.random.choice(neighbors.shape[0], n_approx, replace=True)].values)
        return np.array(sampled_background)
    else:
        sampled_background = {}
        for index, coalition in zip(coalitions_id, coalitions):
            neighbors = conditional.get_neighbors(x, background, coalition, intervals, features_map)
            sampled_background[str(coalition)+str(index)] = neighbors.iloc[np.random.choice(neighbors.shape[0], n_approx, replace=True)].values
        return sampled_background


##coalition sampling

def get_coalitions(L):
    """returns all possible subsets of L"""
    coalitions = []
    for k in range(len(L)+1):
        for coalition in itertools.combinations(L, k):
            coalitions.append(list(coalition))
    return coalitions

def sample_coalitions(p, player_ids, n_coalitions, monte_carlo=True, aggregation=0, constrain=False):
    """returns drawing of coalitions from P(J/j)"""
    features = list(range(p))
    if aggregation == 2: #only 2 players
        others = [feat for feat in features if feat not in player_ids]
        coalitions = [others, []]
    else:
        if aggregation == 0:
            subset = features[:player_ids]+features[player_ids+1:] #J\{j}
        else:
            subset = [feat for feat in features if feat not in player_ids]
        if monte_carlo: #get random coalitions
            if constrain == "fixed":
                if len(subset)>1:
                    coalitions = [list(np.sort(np.random.choice(subset, np.random.randint(1,len(subset)),replace=False))) for k in range(n_coalitions)] + [[],subset.copy()]
                else:
                    coalitions = [[],subset.copy()]
            else:
                coalitions = [list(np.sort(np.random.choice(subset, np.random.randint(0,len(subset)+1),replace=False))) for k in range(n_coalitions)]

        else: #all coalitions without j P(J/j)
            coalitions = get_coalitions(subset)
    return coalitions


def get_coalition_weight(coalition, p, monte_carlo=True, aggregation=0, feature_coalition=None):
    """returns pi(S) or 1 if coalitions are monte carlo sampled"""
    if aggregation == 2:
        return 1/2
    elif monte_carlo:
        return 1
    else:
        if aggregation == 0:
            return 1/(p*int(binom(p-1,len(coalition))))
        else:#aggregation == 1:
            term1 = p-len(feature_coalition)+1
            term2 = int(binom(p-len(feature_coalition),len(coalition)))
            weight = 1/(term1*term2)
            return 1/((p-len(feature_coalition)+1)*int(binom(p-len(feature_coalition),len(coalition))))
    
##computations

def replace_missing(x,coalition,back):
    """returns data with out of coalition replaced with back"""
    p = len(x)
    remaining = [ind for ind in range(p) if ind not in coalition]
    new_x, back = np.array(x), np.array(back)
    new_x[remaining] = back[remaining]
    return new_x


def get_prediction_dataset(X, background, do_print=False, n_coalitions=10, n_approx=1, do_reuse=False, do_double=False, constrain=False, aggregation=0, background_type="random", **kwargs):
    """Returns dataset with random coalitions"""
    
    N, p = X.shape
    feature_names, feature_indexes = list(X.columns), list(range(p))
    
    if aggregation == 0:
        N_shap, shap_names = p, feature_names
    else:
        id_col_mapping = kwargs.get("id_col_mapping")
        assert(id_col_mapping is not None)
        N_shap, shap_names = len(id_col_mapping), list(id_col_mapping.keys())
    
    if background_type == "conditional": #need to divide features into grid to select neighbors
        intervals = kwargs.get("intervals")
        assert(intervals is not None)
    elif background_type == "baseline": #get average values for reference
        baseline = kwargs.get("baseline",None)
        assert(baseline is not None)
    
    if constrain == "sampled" or constrain == "fixed": #constrain empty and full coalitions
        precomputed_coalitions = [{"[]":kwargs.get("background_expectation"), str(feature_indexes):kwargs.get("predictions")[i]} for i in range(N)]
    else:
        precomputed_coalitions = [{} for i in range(N)]
    do_sample_coalitions = (n_coalitions != 0)
    
    sampled_coalitions = [[[] for j in range(N_shap)] for i in range(N)] #sampled coalitions for each feature
    coalitions_status = [[[] for j in range(N_shap)] for i in range(N)]
    prediction_dataset = [] #batch of all predictions for model
    
    if do_print:
        pbar = tqdm(total=N)
    for i, x in enumerate(X.values): #data to explain
        for j, feature in enumerate(shap_names): #features
            
            if aggregation == 0:
                player_id = j
            else:
                player_id = id_col_mapping[feature]
            
            #sample coalitions
            coalitions = sample_coalitions(p, player_id, n_coalitions, monte_carlo=do_sample_coalitions, constrain=constrain, aggregation=aggregation)
            sampled_coalitions[i][j] = coalitions.copy()
            
            
            #filter coalitions for background and computations
            coalitions_to_sample = []
            coalitions_to_sample_id = []
            for k, coalition in enumerate(coalitions):
                coalitions_status[i][j].append([1,1])
                #if do_reuse:
                if str(coalition) in precomputed_coalitions[i].keys():
                    coalitions_status[i][j][k][0] = 3 #already computed
                elif do_reuse:
                    if coalition in coalitions_to_sample:
                        coalitions_status[i][j][k][0] = 2 # already computed in this turn
                    else:
                        coalitions_to_sample.append(coalition)
                        coalitions_to_sample_id.append(k)
                        precomputed_coalitions[i][str(coalition)] = None
                        coalitions_status[i][j][k][0] = 1 #computing + sampling on this turn
                else:
                    coalitions_to_sample.append(coalition)
                    coalitions_to_sample_id.append(k)
                
                if aggregation == 0:
                    coalition_bis = list(np.sort(coalition+[j]))
                else:
                    coalition_bis = list(np.sort(coalition+player_id))
                if str(coalition_bis) in precomputed_coalitions[i].keys():
                    coalitions_status[i][j][k][1] = 3 #already computed
                else:
                    if do_reuse and (coalition_bis in coalitions_to_sample): 
                        coalitions_status[i][j][k][1] = 2 # already computed in this turn
                    else:
                        if do_reuse:
                            precomputed_coalitions[i][str(coalition_bis)] = None
                        if do_double or coalitions_status[i][j][k][0] > 1:
                            coalitions_to_sample.append(coalition_bis)
                            coalitions_to_sample_id.append(k)
                            coalitions_status[i][j][k][1] = 1 #computing + sampling
                        else:
                            coalitions_status[i][j][k][1] = 0 #computing
                
            #sample background                 
            if background_type == "conditional": #on peut optimiser en samplant uniquement du backgronud pour les coalitions pas dans 
                sampled_background = sample_conditional_background(background, x, coalitions_to_sample,  n_approx, intervals, kwargs.get("features_map"), coalitions_to_sample_id)

            elif background_type == "random":
                sampled_background = sample_background(background, coalitions_to_sample, n_approx, coalitions_id = coalitions_to_sample_id)

            #prediction dataset
            for k, coalition in enumerate(coalitions):
                if coalitions_status[i][j][k][0] == 1:
                    if background_type == "baseline":
                        prediction_dataset.append(replace_missing(x,coalition,baseline))
                    else:
                        samples = sampled_background[str(coalition)+str(k)]
                        for sample in samples:
                            prediction_dataset.append(replace_missing(x,coalition,sample))
                
                if coalitions_status[i][j][k][1] <= 1:
                    if aggregation == 0:
                        coalition_bis = list(np.sort(coalition+[j]))
                    else:
                        coalition_bis = list(np.sort(coalition+player_id))                    
                    
                    if background_type == "baseline":
                        prediction_dataset.append(replace_missing(x,coalition_bis,baseline))
                    else:
                        if coalitions_status[i][j][k][1] == 1:
                            samples = sampled_background[str(coalition_bis)+str(k)]
                        elif coalitions_status[i][j][k][0] == 1:
                            samples = sampled_background[str(coalition)+str(k)]
                        else:
                            raise ValueError("BUG in monte_carlo prediction_dataset : Impossible outcome")
                        for sample in samples:
                            prediction_dataset.append(replace_missing(x,coalition_bis,sample))
        if do_print:
            pbar.update(1)
            
    outputs = {"prediction_dataset":pd.DataFrame(np.array(prediction_dataset),columns=feature_names), "sampled_coalitions":sampled_coalitions, "precomputed_coalitions":precomputed_coalitions, "coalitions_status": coalitions_status}
    return outputs



def delta_v(v1, v2, weight):
    """returns approximation of delta_v using v1 and v2 preds"""
    return weight * (v2-v1)

                                     
def compute_shapley(X, outputs, n_coalitions=10, n_approx=1, do_reuse=False, aggregation=0, do_print=False, **kwargs):
    
    N, p = X.shape
    feature_names, feature_indexes = list(X.columns), list(range(p))
    if aggregation == 0:
        N_shap, shap_names = p, feature_names
    else:
        id_col_mapping = kwargs.get("id_col_mapping")
        assert(id_col_mapping is not None)
        N_shap, shap_names = len(id_col_mapping), list(id_col_mapping.keys())
    do_sample_coalitions = (n_coalitions != 0)

    sampled_coalitions, precomputed_coalitions, coalitions_status, coalitions_preds = outputs["sampled_coalitions"], outputs["precomputed_coalitions"], outputs["coalitions_status"], outputs["coalitions_preds"]

    assert(not np.isnan(coalitions_preds).any())
    
    shapley_values = {feature:np.zeros(N) for feature in shap_names}
    pos = 0
    if do_print:
        pbar = tqdm(total=N)
    for i in range(N):
        for j, feature in enumerate(shap_names):
            
            if aggregation == 0:
                player_id = j
            else:
                player_id = id_col_mapping[feature]
            coalitions = sampled_coalitions[i][j]
            
            coalition_values = []
            for k, coalition in enumerate(coalitions):
                if aggregation == 0:
                    coalition_bis = list(np.sort(coalition+[j]))
                else:
                    coalition_bis = list(np.sort(coalition+player_id))
                coalition_status = coalitions_status[i][j][k][0]
                coalition_status_bis = coalitions_status[i][j][k][1]
                
                #v(S)
                if coalition_status>=2:
                    coalition_pred = precomputed_coalitions[i][str(coalition)]
                else:
                    coalition_pred = np.mean(coalitions_preds[pos:pos+n_approx])
                    if np.isnan(coalition_pred):
                        raise ValueError("DEBUG", pos, n_approx, coalition, len(coalitions_preds), coalitions_preds[pos:pos+n_approx])
                    pos += n_approx
                    if do_reuse:
                        precomputed_coalitions[i][str(coalition)] = coalition_pred
                
                #v(S+j)
                if coalition_status_bis>=2:
                    coalition_pred_bis = precomputed_coalitions[i][str(coalition_bis)]              
                else:                    
                    coalition_pred_bis = np.mean(coalitions_preds[pos:pos+n_approx])
                    pos += n_approx
                    if do_reuse:
                        precomputed_coalitions[i][str(coalition_bis)] = coalition_pred_bis

                weight = get_coalition_weight(coalition, p, monte_carlo=do_sample_coalitions, aggregation=aggregation, feature_coalition=player_id)
                difference = delta_v(coalition_pred, coalition_pred_bis, weight)
                coalition_values.append(difference) 
            
            if aggregation==2 or not do_sample_coalitions:
                shapley_values[feature][i] = np.sum(coalition_values)
            else:
                shapley_values[feature][i] = np.mean(coalition_values)
                
        if do_print:
            pbar.update(1)
        outputs["precomputed_coalitions"] = precomputed_coalitions
    return pd.DataFrame(shapley_values)

    
                                             
         