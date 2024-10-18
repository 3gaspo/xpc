import pandas as pd


def get_interval_length(background, col, N):
    """returns interval length for grid of size N for col"""
    col_min, col_max = background[col].min(), background[col].max()
    return (col_max - col_min) / N

def get_continuous_neighbors(x, background, col_id, distance):
    """returns neighbors of non-cyclic continuous feature at col_id based on distance"""
    col = background.columns[col_id]
    a, b = x[col_id]-distance/2, x[col_id]+distance/2
    neighbors = background[background[col].between(a,b)]
    return neighbors

def get_cyclic_neighbors(x, background, col_id, distance):
    """returns neighbors of cyclic feature at col_id based on distance"""
    col = background.columns[col_id]
    m, M = background[col].min(), background[col].max()
    a, b = x[col_id]-distance/2, x[col_id]+distance/2
    
    abis, bbis = None, None
    if a<m:
        abis = M-(m-a)
        a = m
    if b>M:
        bbis = m + (b-M)
        b = M
    mask = background[col].between(a,b)
    if abis is not None:
        mask = mask | background[col].between(abis,M)
    if bbis is not None:
        mask = mask | background[col].between(m,bbis)
    return background[mask]

def get_categorical_neighbors(x, background, col_id):
    """returns neighbors for categorical feature"""
    col = background.columns[col_id]
    value = x[col_id]
    neighbors = background[background[col] == value]
    return neighbors


def get_intervals(background, n_intervals, features_map):
    if n_intervals == 0 or n_intervals==1:
        return {col:None for col in background}        
    else:
        return {col: 0 if features_map[col] == "categorical" else get_interval_length(background, col, n_intervals) for col in background}

    
def get_neighbors(x, background, col_ids, intervals, features_map):
    """returns neighbors of x in background based on features length grid"""
    columns = list(background.columns)
    if len(col_ids) == len(x):
        return pd.DataFrame([x],columns=columns)
    else:
        neighbors = background.copy().reset_index(drop=True)
        for col_id in col_ids:
            col = columns[col_id]
            interval_length = intervals[col]
            if interval_length is not None:
                if features_map[col] == "categorical":
                    temp_df = get_categorical_neighbors(x, neighbors, col_id)
                elif features_map[col] == "continuous":
                    temp_df = get_continuous_neighbors(x, neighbors, col_id, interval_length)
                else:
                    temp_df = get_cyclic_neighbors(x, neighbors, col_id, interval_length)
                if not temp_df.empty:
                    neighbors = temp_df.copy()
        return neighbors