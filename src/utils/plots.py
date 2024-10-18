import matplotlib.pyplot as plt



def show_predictions(y, predictions=None, week=True, show_days=True, labels=None, title="Temporal evolution", Nweek=336, Nday=48, x=None):
    """plots y and predictions if specified"""

    fig = plt.figure(figsize=(12,3))
    if week:
        y = y.iloc[:336]
        if predictions is not None:
            predictions = predictions.iloc[:336]
    
    if x is None:
        xx = range(y.shape[0])
    else:
        xx = x
        
    plt.plot(xx, y, label="True values", color="limegreen")
    if predictions is not None:
        plt.plot(xx, predictions, label="Predicted", color="tomato")

    if show_days:
        for k in range(0,y.shape[0],Nday):
            plt.axvline(x=k,linestyle="--")
    
    if x is None:
        plt.xticks([])
    
    if title is None:
        title = "Temporal evolution"
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.show();

    
def show_sum(df, offset=0, true_values=None, show_days=True, week=True, title="Decomposition of the time serie", Nweek=336, Nday=48, x=None, color_mapping=None):
    """plots additive decomposition from df"""
    if week:
        df = df.iloc[:Nweek]
    if x is None:
        x = range(df.shape[0])
        
    fig = plt.figure(figsize=(12,3))
    if true_values is not None:
        plt.plot(x, true_values.reset_index(drop=True), "--", label="True", linewidth=0.5)
    
    serie = offset

    for col in df.columns:
        new_serie = df[col].reset_index(drop=True)
        if color_mapping:
            plt.fill_between(x=x, y1=serie, y2 =serie+new_serie, label=col, color=color_mapping[col])
        else:
            plt.fill_between(x=x, y1=serie, y2 =serie+new_serie, label=col)
        serie = serie + new_serie
        
    if show_days:
        for k in range(0, df.shape[0], Nday):
            plt.axvline(x=k,linestyle="--")
    
    if title is None:
        title = "Decomposition of the time series"
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.show();
    
    
    
def show_separate(df, offset=0, show_days=True, week=True, title="Decomposition of the time serie", Nweek=336, Nday=48, x=None,color_mapping=None):
    """plots each decomposition from df"""
    if week:
        df = df.iloc[:Nweek]

    fig = plt.subplots(1,df.shape[1],figsize=(12,3))

    if x is None:
        x = range(df.shape[0])
    
    for i,col in enumerate(df.columns):
        new_serie = df[col].reset_index(drop=True)
        if color_mapping:
            ax[i].plot(x, new_serie, label=col, color=color_mapping[col])
        else:
            ax[i].plot(x, new_serie, label=col)

        if show_days:
            for k in range(0, df.shape[0], Nday):
                ax[i].axvline(x=k,linestyle="--")
        ax[i].legend()
    
    if title is None:
        title = "Decomposition of the time series"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show();