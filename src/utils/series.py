import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def get_col_weights(col_mapping):
    """computes weights for each col. If col is in multiple parts, its weight decreases"""
    col_weights = {}
    for part_name in col_mapping:
        for col in col_mapping[part_name]:
            if col in col_weights:
                col_weights[col] += 1
            else:
                col_weights[col] = 1
    return {key:1/value for key,value in col_weights.items()}
    
def aggregate_series(df, col_mapping):
    """merges cols according to col_mapping (sum)"""
    merge_df = pd.DataFrame()
    added_cols = []
    col_weights = get_col_weights(col_mapping)

    for part_name in col_mapping:
        cols = [col for col in col_mapping[part_name] if col in df.columns]
        added_cols += cols
        filtered_col_weights = {key:value for key,value in col_weights.items() if key in cols}
        merge_df[part_name] = (df[cols] * filtered_col_weights).sum(axis=1)

    #multivariate cols
    for col in df.columns.drop(added_cols):
        cols = col.split(",")
        N = len(cols)
        for part_name in col_mapping:
            new_cols = [col for col in cols if col in col_mapping[part_name]]
            merge_df[part_name] = merge_df[part_name] + df[new_cols].sum(axis=1) * (len(new_cols)/N)
    return merge_df

def get_percents(df, div_values=None, percents=False):
    """express df as percentages"""
    if div_values is None:
        _percents = df.div(df.sum(axis=1),axis=0)
    elif isinstance(div_values, float):
        _percents = (df / div_values)
    else:
        _percents = df.div(div_values.values, axis=0)
    if percents:
        _percents = 100 * _percents
    return _percents

def get_deltas(df, df_mins=None, do_all=True, alpha=1):
    """express df as different to minimums if negative.
    Do_all=False:heighten only negative.
    Alpha:multiplicative weight for infs (e.g 1.1)"""
    if df_mins is None: #compute minimums of df directly
        df_mins = df.min(axis=0)
    if do_all:
        delta_df = df.apply(lambda col: col - alpha*min(0, df_mins[col.name]), axis=0)
        min_sum = alpha * df_mins[df_mins < 0].sum()
    else:
        delta_df = df.apply(lambda col: col - alpha*df_mins[col.name], axis=0)
        min_sum = alpha * df_mins.sum()
        
    return delta_df, df_mins, min_sum

def get_errors(predicted_df,true_df, relative=False, percents=False, absolute=False):
    if not isinstance(predicted_df, pd.Series):
        predicted_df = pd.Series(predicted_df)
    if relative:
        errors = ((predicted_df.reset_index(drop=True) - true_df.reset_index(drop=True)) / true_df.reset_index(drop=True))
    else:
        errors = predicted_df.reset_index(drop=True) - true_df.reset_index(drop=True)
    if percents:
        errors = 100*errors
    if absolute:
        errors = errors.abs()
    return errors

def score(predicted_df,true_df, relative=False, percents=False, absolute=False):
    """returns (relative) error (in percents)"""
    errors = get_errors(predicted_df,true_df, relative=relative, percents=percents, absolute=absolute)
    return errors.mean(axis=0)


def time_model(model, X):
    """computes models predictions and returns predicitons & computation times"""
    t1 = perf_counter()
    predictions = model.predict(X)
    t2 = perf_counter()
    return predictions, t2-t1

def analyze_errors(predicted_df,true_df, relative=False, percents=False, absolute=False, title="Errors"):
    plt.figure(figsize=(6,2))
    errors = get_errors(predicted_df,true_df, relative=relative, percents=percents, absolute=absolute)
    print(f"Mean {title} : {np.mean(errors):.2f} %")
    errors.plot.hist(bins=100)
    plt.title(title)
    if percents:
        plt.xlabel('Errors %')
    else:
        plt.xlabel('Errors')
    plt.ylabel('Counts')
    plt.show()    