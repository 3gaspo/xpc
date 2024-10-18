import numpy as np
import pandas as pd
from time import perf_counter
from tqdm.notebook import tqdm
#from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import copy


from xpc.shapley import preprocess
from xpc.shapley import explainers
from xpc.utils import series

    
def pipeline(model, background, X, algo, background_dict={}, X_dict={}, col_mapping=None, features=None, return_all=False,  **kwargs):
    t0 = perf_counter()
    explainer = explainers.Shapley(model, algo=algo, col_mapping=col_mapping, **kwargs)
    explainer.fit(background, background_dict, features=features, **kwargs)
    parts = explainer(X, X_dict, **kwargs)
    errors_dict = explainer.get_simple_errors()
    tend = perf_counter()
    print("")
    print("============================================")
    print(f"Total computations done in {(tend-t0)/60:.2f} min")
    if return_all:
        return parts, errors_dict, explainer
    else:
        return parts
    
def get_errors_stats(experiment_dict):
    """returns error means and vars"""
    
    param_sets = experiment_dict[next(iter(experiment_dict.keys()))].keys()
    K =  len(experiment_dict[next(iter(experiment_dict.keys()))][next(iter(param_sets))]) #len(param_range)
    repetitions = len(experiment_dict[next(iter(experiment_dict.keys()))][next(iter(param_sets))][0])
    
    errors_smach, errors_spline, errors_exact, errors_efficiency, times = experiment_dict["errors_smach"], experiment_dict.get("errors_spline"), experiment_dict.get("errors_exact"), experiment_dict["errors_efficiency"], experiment_dict["times"]
    
    mean_times = {key:[] for key in param_sets}
    mean_errors_smach = {key:[] for key in param_sets}
    mean_errors_spline = {key:[] for key in param_sets}
    mean_errors_exact = {key:[] for key in param_sets}
    mean_errors_efficiency = {key:[] for key in param_sets}
    var_times = {key:[] for key in param_sets}
    var_errors_smach = {key:[] for key in param_sets}
    var_errors_spline = {key:[] for key in param_sets}
    var_errors_exact = {key:[] for key in param_sets}
    var_errors_efficiency = {key:[] for key in param_sets}    

    for param_ind in param_sets:
        for i in range(K):
            var_errors_smach[param_ind].append(np.std(errors_smach[param_ind][i]))
            var_errors_efficiency[param_ind].append(np.std(errors_efficiency[param_ind][i]))
            var_times[param_ind].append(np.std(times[param_ind][i]))
            if errors_exact is not None:
                var_errors_exact[param_ind].append(np.std(errors_exact[param_ind][i]))
            if errors_spline is not None:
                var_errors_spline[param_ind].append(np.std(errors_spline[param_ind][i]))

            mean_errors_smach[param_ind].append(np.mean(errors_smach[param_ind][i]))
            mean_errors_efficiency[param_ind].append(np.mean(errors_efficiency[param_ind][i]))
            mean_times[param_ind].append(np.mean(times[param_ind][i]))
            if errors_exact is not None:
                mean_errors_exact[param_ind].append(np.mean(errors_exact[param_ind][i]))
            if errors_spline is not None:
                mean_errors_spline[param_ind].append(np.mean(errors_spline[param_ind][i]))


    final_dict = {"times":mean_times, "errors_smach":mean_errors_smach, "errors_efficiency":mean_errors_efficiency, "var_times":var_times, "var_errors_smach":var_errors_smach, "var_errors_efficiency":var_errors_efficiency}
    if errors_exact is not None:
        final_dict["errors_exact"] = mean_errors_exact
        final_dict["var_errors_exact"] = var_errors_exact
    if errors_spline is not None:
        final_dict["errors_spline"] = mean_errors_spline
        final_dict["var_errors_spline"] = var_errors_spline
    return final_dict


def filter_days(dico, ind, Nx):
    new_dico = {}
    for key in dico:
        if type(dico[key]) in [pd.Series, pd.DataFrame]:
            new_dico[key] = dico[key].copy().reset_index(drop=True).iloc[ind:ind+Nx]
        else:
            new_dico[key] = dico[key].copy()
    return new_dico
    
    
def do_experiment(background, X, param_sets, param_ranges={}, repetitions=1, col_mapping=None, Nx=None, background_dict={}, X_dict={}, do_exact=False, unique_model=True, model=None, save=True, expe_title="lastest_expe", folder="latest_results", **kwargs):
    """experiment on each param_set along param_to_test"""
    
    
    S = len(param_sets)
    if param_ranges != {}:
        K = len(param_ranges[next(iter(param_ranges.keys()))])
    else:
        K = 1
    times = {key:[[] for k in range(K)] for key in param_sets}
    errors_smach = {key:[[] for k in range(K)] for key in param_sets}
    errors_spline = {key:[[] for k in range(K)] for key in param_sets}
    errors_exact = {key:[[] for k in range(K)] for key in param_sets}
    errors_efficiency = {key:[[] for k in range(K)] for key in param_sets}
    total = S * repetitions * K
    
    if unique_model:
        assert(model is not None and hasattr(model, "MyModelMarker"))
        background_dictt = preprocess.get_background_data(model, background, col_mapping=col_mapping, **background_dict, **kwargs)
        X_dictt = preprocess.get_X_data(model, X, background, background_dict=background_dict, col_mapping=col_mapping, do_exact=do_exact, **X_dict, **kwargs)
    
    else:
        background_dictt = background_dict
        X_dictt = X_dict

    
    pbar = tqdm(total = total)
    t0 = perf_counter()
    for rep in range(repetitions):
        print(f"Repetition {rep+1}")
        
        if Nx is not None:
            ind = np.random.randint(X.shape[0]-Nx)
            _X = X.reset_index(drop=True).iloc[ind:ind+Nx]
            background_dicttt = filter_days(background_dictt, ind, Nx)
            X_dicttt = filter_days(X_dictt, ind, Nx)
        else:
            _X = X.copy()
            background_dicttt = _background_dictt.copy()
            X_dicttt = _X_dictt.copy()
        
        for s, param_set_name in enumerate(param_sets):
            param_set = copy.deepcopy(param_sets[param_set_name])
            algo = param_set["algo"]
            if not unique_model:
                model = param_set["model"]
                print("loading model:",model.model_class, model.features)
            params = {key:value for key,value in param_set.items() if key not in ["algo","model"]}
            
            if not unique_model:
                background_dictttt = preprocess.get_background_data(model, background, col_mapping=col_mapping, **background_dicttt, **kwargs)
                X_dictttt = preprocess.get_X_data(model, _X, background, background_dict=background_dictttt, col_mapping=col_mapping, do_exact=do_exact, **X_dicttt, **kwargs)
            else:
                background_dictttt = background_dicttt
                X_dictttt = X_dicttt
                
            for i in range(K):
                if K>1:
                    for param_name in param_ranges:
                        params[param_name] = param_ranges[param_name][i]
            
                t1 = perf_counter()
                explainer = explainers.Shapley(model, algo=algo, col_mapping=col_mapping, **params)
                explainer.fit(background, background_dictttt, **params)
                parts = explainer(_X, X_dictttt, **params)
                errors_dict = explainer.get_simple_errors()
                t2 = perf_counter()
                
                errors_smach[param_set_name][i].append(errors_dict["error_smach_abs"])
                errors_efficiency[param_set_name][i].append(errors_dict["error_efficiency_abs"])
                times[param_set_name][i].append(t2-t1)
                if model.model_class == "gam":
                    errors_spline[param_set_name][i].append(errors_dict["error_spline_abs"])
                if do_exact:
                    errors_exact[param_set_name][i].append(errors_dict["error_exact_abs"])

                pbar.update(1)

    tend = perf_counter()
    print("")
    print("============================================")
    print(f"Total computations done in {(tend-t0)/60:.2f} min")
    
    experiment_dict = {"times":times,"errors_efficiency":errors_efficiency, "errors_smach":errors_smach, "errors_spline":errors_spline, "errors_exact":errors_exact}
    if save:
        with open(f'{folder}/{expe_title}.json', 'w') as f:
            json.dump(experiment_dict, f)
    return experiment_dict

                       

titles = {"errors_efficiency":"Efficiency error","errors_smach":"SMACH error", "errors_exact":"SHAP error","errors_spline":"GAM error" ,"times":"Computation time"}
y_labels = {"errors_efficiency":"Error (mean %)","errors_smach":"Error (mean %)", "errors_exact":"Error (mean %)", "errors_spline":"Error (mean %)","times":"Time (s)"}

    
def plot_all(experiment_dict, param_ranges, param_name="Parameters"):
    """plots all the repetitions"""
    
    param_sets = experiment_dict[next(iter(experiment_dict.keys()))].keys()
    repetitions = len(experiment_dict[next(iter(experiment_dict.keys()))][next(iter(param_sets))][0])
    
    fig, axs = plt.subplots(len(param_sets), len(experiment_dict), figsize=(5*len(experiment_dict), 5*len(param_sets)))    
    
    do_logs=True
    if len(param_ranges)>1:
        param_range = range(len(param_ranges))
        do_logs = False
    else:
        param_range = param_ranges[next(iter(param_ranges.keys()))]
        
    errors_dict = get_errors_stats(experiment_dict)
    for k,key in enumerate(experiment_dict):
        for j, param_ind in enumerate(param_sets):
            experiment = np.array(experiment_dict[key][param_ind])
            if experiment.shape[1]>0:
                if len(param_sets)>1:
                    ax = axs[j][k]
                else:
                    ax = axs[k]

                if repetitions>1:
                    ax.plot(param_range, errors_dict[key][param_ind], label=param_ind)
                    for i in range(repetitions):
                        ax.plot(param_range, experiment[:,i], alpha=0.2)
                else:
                    ax.plot(param_range, experiment[:,0], label=param_ind)
            ax.set_title(titles[key])
            ax.set_ylabel(y_labels[key])
            ax.set_xlabel(param_name)
            ax.legend()
            if do_logs:
                ax.set_xscale("log")
                
    fig.tight_layout()
    plt.show();
          
        
def plot_stats(experiment_dict, param_range, param_name, errors=True):
    """plots all the repetitions"""
    
    param_sets = experiment_dict[next(iter(experiment_dict.keys()))].keys()
    repetitions = len(experiment_dict[next(iter(experiment_dict.keys()))][next(iter(param_sets))][0])
    
    fig, axs = plt.subplots(1, len(experiment_dict), figsize=(5*len(experiment_dict), 5))    
    
    do_logs=True
    if len(param_range.shape)>1:
        param_range = range(len(param_range))
        do_logs = False
        
    errors_dict = get_errors_stats(experiment_dict)
    for k,key in enumerate(experiment_dict):
        for j, param_ind in enumerate(experiment_dict[key]):
            experiment = np.array(experiment_dict[key][param_ind])
            ax = axs[k]
                
            if errors:
                ax.errorbar(param_range, errors_dict[key][param_ind], np.array(errors_dict["var_"+key][param_ind]),label=param_ind, capsize=3)
            else:
                ax.plot(param_range, errors_dict[key][param_ind],label=param_ind)
                
            ax.set_title(titles[key])
            ax.set_ylabel(y_labels[key])
            ax.set_xlabel(param_name)
            ax.legend()
            if do_logs:
                ax.set_xscale("log")
                
    fig.tight_layout()
    plt.show();
        
def squeeze(dico, ind=-1):
    """removes param_range dimension from dictionnary"""
    return {key: {keyy:np.array(dico[key][keyy])[ind,:] for keyy in dico[next(iter(dico.keys()))].keys()} for key in dico.keys()}
    
def plot_boxes(experiment_dict):
    """plots boxplots"""
    param_sets = experiment_dict[next(iter(experiment_dict.keys()))].keys()
    repetitions = len(experiment_dict[next(iter(experiment_dict.keys()))][next(iter(param_sets))][0])
    fig, axs = plt.subplots(1, len(experiment_dict), figsize=(5*len(experiment_dict), 5))   
    experiment_dict = squeeze(experiment_dict)
    for k,key in enumerate(experiment_dict):
        ax = axs[k]
        for j, param_ind in enumerate(experiment_dict[key]):
            ax.boxplot(experiment_dict[key].values(), labels = experiment_dict[key].keys())
            ax.tick_params(labelrotation=90)
            ax.set_title(titles[key])
            ax.set_ylabel(y_labels[key])
            ax.set_xlabel("Experiments")
                
    fig.tight_layout()
    plt.show();
     
        
y_labels = {"errors_efficiency":"Efficiency (%)","errors_smach":"SMACH error (%)", "errors_exact":"SHAP error (%)", "errors_spline":"GAM error (%)","times":"Time (s)"}
            
            
def print_results_table(experiment_dict, index=None):
    """prints results of experiments at given param"""
    
    param_sets = experiment_dict[next(iter(experiment_dict.keys()))].keys()    
    errors_dict = get_errors_stats(experiment_dict)
    K = len(errors_dict[next(iter(errors_dict.keys()))][next(iter(param_sets))])

    
    header = "Experiment".ljust(20) + "".join([y_labels[metric].ljust(20) for metric in experiment_dict])
    print(header)
    print("-" * len(header))
    
    if index is None:
        if K == 1:
            index = 0
        else:
            index = -1
    
    for exp_id in param_sets:
        row = exp_id.ljust(20)
        for metric in experiment_dict:
            row += str(round(errors_dict[metric][exp_id][index],2)).ljust(20)
        print(row)
  
        
def score_models(models_dict, X_train, y_train, X_test, y_test):
    """Scores models and prints results in a formatted table."""
    
    header = "Model".ljust(20) + "Train Error (%)".ljust(20) + "Test Error (%)".ljust(20) + "Time (min)".ljust(20)
    print(header)
    print("-" * len(header))
    
    for model_name, model in models_dict.items():
        t1 = perf_counter()
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)
        error_train = series.score(preds_train, y_train, relative=True, percents=True, absolute=True)
        error_test = series.score(preds_test, y_test, relative=True, percents=True, absolute=True)
        t2 = perf_counter()
        
        row = model_name.ljust(20)
        row += str(round(error_train, 2)).ljust(20)
        row += str(round(error_test, 2)).ljust(20)
        row += f"{(t2-t1)/60:.3f}".ljust(20)
        print(row)
