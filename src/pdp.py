import shap

def draw(model, X, feature, npoints=10):
    shap.plots.partial_dependence(feature, model.predict, X, npoints=npoints)