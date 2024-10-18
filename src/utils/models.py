import pandas as pd
import numpy as np

class MyModel:
    """model class to enable prediction, from a model class object, prediction function, or Rscript"""
    
    def __init__(self, model, **kwargs):
        """path_kw used only if model is path string (Rscript), for data path"""
        self.MyModelMarker = True
        
        if isinstance(model,str): #R model as a folder containing .rds model and script
            self.model_type = "script"
            self.model_path = model
            self.script_name = kwargs.get("script_name", "script.R")
            self.predict_fct = self._make_script_pred
            
        elif callable(model): #predict function
            self.model_type = "predict"
            self.predict_fct = model
        elif hasattr(model,"predict") and callable(model.predict): #model class
            self.model_type = "class"
            self.predict_fct = model.predict #must already be fitted
        else:
            raise ValueError('Please provide a correct model : path to R script, python class or predict function')
            
        self.model_class = kwargs.get("model_class",None) #gam, dragon, ...
        self.features = kwargs.get("features",None) #if model takes subset of features as input
            
    def _make_script_pred(self, data, debug=False, **kwargs):
        """saves data to csv, executes script at model_path, and returns pred from saved pred csv"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.model_path + "/" + "input.csv")
        elif isinstance(data, np.ndarray):
            if self.features is not None:
                features = self.features
            elif kwargs.get("features") is not None:
                features = kwargs.get("features")
            else:
                raise ValueError("Please provide features for the array")
            data = pd.DataFrame(data,columns=features)
            data.to_csv(self.model_path + "/" + "input.csv")
        else:
            raise ValueError("Please provide a dataframe or array+features")
        
        full_model_math = self.model_path + '/' + self.script_name
        if debug:
            print("")
            print("Loading model script at:",self.model_path + '/' + self.script_name)
            print("--------------------------------------")
            get_ipython().system("R < {full_model_math} --vanilla")
            print("--------------------------------------")
            print("Done script pred")
            print("")
        else: #no printing from R script
            get_ipython().system('R < {full_model_math} --vanilla >/dev/null 2>&1') 
        
        preds = pd.read_csv(self.model_path + "/" + "output.csv")
        if preds.shape[0] != data.shape[0]:
            raise ValueError("pred is not the same size as input, see debug in script model")
        return preds
    
    
    def predict(self, data, **kwargs):
        """returns predictions"""
        
        if self.model_type == "script":
            preds = self.predict_fct(data, **kwargs)
            
            if self.model_class == "gam":
                if kwargs.get("return_all",False):
                    cols = [col for col in preds.columns if "pred_" in col] + ["EstimatedLoad"]
                    preds = preds[cols]
                    preds = preds.rename(columns={col:col.replace("pred_","").replace("s(","").replace("te(","").replace(")","").replace("as.factor(","").replace("offset:","") for col in preds.columns})
                    for col in data.columns:
                        if col not in preds.columns:
                            preds[col] = np.zeros(preds.shape[0])
                    return preds
                else:
                    return preds["EstimatedLoad"]
            else:
                return preds
            
        else:
            preds = self.predict_fct(data, **kwargs)
            if not isinstance(preds, pd.Series):
                preds = pd.Series(preds)
            return preds
