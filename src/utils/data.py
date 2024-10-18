import pandas as pd
import numpy as np
import json


## dataframes

def filter_date(df, years=None, date_column="date"):
    """changes date to datetime format and filters df by specified years, sets index as data"""
    new_df = df.copy()
    if date_column in df.columns:
        new_df[date_column] = pd.to_datetime(new_df[date_column])
    else:
        try:
            new_df[date_column] = pd.to_datetime(new_df.index)
        except:
            raise ValueError("Couldn't find dates in columns or index")
            
    if years is None:
        years = np.unique(new_df[date_column].dt.year)
    new_df = new_df[new_df[date_column].dt.year.isin(years)]
    new_df = new_df.set_index(date_column)

    return new_df


def read_dataset(name, path=None, years=None, cols_to_drop=None, rename_cols=None, date_column="date"):
    """returns dataframe form named csv"""
    if path is not None:
        df = pd.read_csv(f"{path}/{name}.csv")
    else:
        df = pd.read_csv(f"{name}.csv")
    
    df = filter_date(df,years,date_column)
    
    if cols_to_drop is not None:
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
    if rename_cols is not None:
        df = df.rename(columns = rename_cols)
        
    return df


def wrap_data(data,features=None, y_name="Load", index=None):
    """wraps input data as dataframe or dataloader"""
        
    if isinstance(data, pd.DataFrame):
        if features is not None:
            data = data[features]
        if y_name in data.columns:
            data = data.drop(columns=[y_name])
    elif isinstance(data, np.ndarray) and features is not None:
        if data.shape[1] != len(features):
            raise ValueError("Not enough features for given array size")
        data = pd.DataFrame(data,columns=features)
    else:
        raise TypeError("Please provide a dataframe or array and features")
    if index is not None:
        data.index = index
    return data


def get_median(df):
    """returns median vector of df"""
    median_vector = []
    for col in df.columns:
        sorted_values = df[col].sort_values()
        n = len(sorted_values)
        median_vector.append(sorted_values.iloc[n // 2])
    return median_vector


## dictionaries

def open_json(name,folder=None):
    if folder:
        path = f'{folder}/{name}.json'
    else:
        path = f'{name}.json'
    with open(path, 'r') as f:
        loaded_data = json.load(f)    
    return loaded_data


def filter_keys(dico, new_keys, nested=True):
    """filters keys of dictionnary""" 
    if not nested:
        new_dico =  {new_key: dico[new_key]  for new_key in new_keys}
    else:
        new_dico =  {meta_key: {key:dico[meta_key][key] for key in new_keys} for meta_key in dico.keys()}
    return new_dico

def filter_index(dico, new_start=None, new_end=None, nested=True):
    """filters keys of dictionnary""" 
    if not nested:
        new_dico =  {key: dico[new_key][new_start:new_end]  for key in dico.keys()}
    else:
        new_dico =  {meta_key: {key:dico[meta_key][key][new_start:new_end] for key in dico[next(iter(dico.keys()))].keys()} for meta_key in dico.keys()}
    return new_dico

def filter_mapping(mapping, keep):
    """filters mapping of lists with only values in keep"""
    new_mapping = {}
    for map_name in mapping.keys():
        _map = mapping[map_name]
        _new_map = []
        for value in _map:
            if value in keep:
                _new_map.append(values)
        new_mapping[map_name] = _new_map
    return new_mapping



def open_json(name,folder=None):
    if folder:
        path = f'{folder}/{name}.json'
    else:
        path = f'{name}.json'
    with open(path, 'r') as f:
        loaded_data = json.load(f)    
    return loaded_data
    