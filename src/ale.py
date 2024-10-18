from PyALE import ale
import matplotlib.pyplot as plt

def draw_1d(model, X, features, feature_type="continuous", include_CI=False, grid_size=50):
    if type(features) is list:
        fig, ax = plt.subplots(1,len(features), figsize=(15,4))
        for i,feature in enumerate(features):
            ale(X=X, model=model, feature=[feature], feature_type=feature_type, include_CI=include_CI, grid_size=grid_size, fig=fig, ax=ax[i])
    else:
        ale(X=X, model=model, feature=[features], feature_type=feature_type, include_CI=include_CI, grid_size=grid_size)
    
def draw_2d(model, X, features, feature_type="continuous", include_CI=False, grid_size=50):
    if type(features) is not list or len(features) != 2:
        raise TypeError("Please provide list of 2 features")
    ale(X=X, model=model, feature=features, feature_type=feature_type, include_CI=include_CI, grid_size=grid_size)