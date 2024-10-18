import numpy as np
import pandas as pd
from time import perf_counter
from tqdm.notebook import tqdm

import shap
from xpc.utils import models
from xpc.utils import series
from xpc.utils import plots
from xpc.utils import data
from xpc.shapley import monte_carlo
from xpc.shapley import conditional
from xpc.shapley import kernel
from xpc.shapley import preprocess
from xpc.shapley import gams

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)


        
class DefaultExplainer:
    def __init__(self, predict_fct, do_print=False): 
        self.predict_fct = predict_fct
        self.do_print = do_print
        
    def fit(self):
        pass
    
    def __call__(self):
        pass

    def print_params(self, params):
        """prints string of params from dict"""
        string = ""
        i, N = 1, len(params)
        for key,value in params.items():
            if i<N:
                string += f" {key}={value} |"
            else:
                string += f" {key}={value} -"
            i+=1
        print(string,end=" ")
    
    
class PermutationExplainer(DefaultExplainer):
    """PermutationSHAP explainer using shap library"""
    def __init__(self, predict_fct, do_print=False, n_permutations=10, **kwargs):
        super().__init__(predict_fct, do_print=do_print)
        self.n_permutations = n_permutations
        self.relevant_params = {"n_permutations":n_permutations}
        
    def fit(self, background, background_dict, **kwargs):
        self.expected_value = background_dict["background_expectation"]
        self.max_evals = self.permutations * (2*background.shape[1]+1) #calls to the model (shap uses antithetic sampling)
        self.relevant_params["max_evals"] = self.max_evals
        if self.do_print: self.print_params(self.relevant_params)
        self.explain_fct = shap.Explainer(self.predict_fct, background, algorithm="permutation")
        
    def __call__(self, X, **kwargs):
        shap_explanation = self.explain_fct(X, max_evals=self.max_evals)
        shap_expected_value = shap_explanation.base_values[0]
        assert(shap_expected_value == self.expected_value)
        shap_values = pd.DataFrame(shap_explanation.values, columns=X.columns, index=X.index)
        return shap_values
        
        
class KernelExplainer(DefaultExplainer):
    """KernelSHAP using shap library or exact regression"""
    def __init__(self, predict_fct, do_print=False, do_exact=False, **kwargs):
        super().__init__(predict_fct, do_print=do_print)
        self.do_exact = do_exact
        self.reference = kwargs.get("reference","median") #for exact kernel
        self.max_evals = kwargs.get("max_evals","auto") #for shap kernel
        self.relevant_params = {"do_exact":do_exact}
    
    def fit(self, background, background_dict, **kwargs):
        self.expected_value = background_dict["background_expectation"]
        if self.do_exact:
            if self.model_type == "dragon":
                raise TypeError("DEBUG expected value not yet supported for dragon")
            else:
                if self.reference == "median":
                    self.reference_vector = data.get_median(background)
                elif self.reference == "expectation":
                    self.reference_vector = np.mean(background,axis=0)
                elif self.reference == "origin":
                    self.reference_vector = np.array([0 for k in range(background.shape[1])])
                else:
                    self.reference_vector = None
                if reference_vector is not None:
                    self.expected_value = self.predict_fct(pd.DataFrame([self.reference_vector],columns=background.columns)).values[0]
            self.relevant_params["reference"] = self.reference
        else:
            self.explain_fct = shap.KernelExplainer(self.predict_fct, background)
            self.relevant_params["max_evals"] = self.max_evals
        if self.do_print: self.print_params(self.relevant_params)

    def __call__(self, X, X_dict={}, **kwargs):
        if self.do_exact:
            shap_explanation = kernel.exact_regression(X, self.predict_fct, reference=self.reference_vector, do_print=self.do_print)
        else:
            shap_explanation = self.explain_fct.shap_values(X, nsamples = self.max_evals)
            shap_expected_value = self.explain_fct.expected_value
            assert(shap_expected_value == self.expected_value)
        shap_values = pd.DataFrame(shap_explanation, columns = self.X.columns)
        return shap_values
        


class ExactExplainerGAM(DefaultExplainer):
    """exact explainer for monovariate GAM"""
    def __init__(self, predict_fct, do_print=False, n_coalitions=10, n_approx=1, **kwargs):
        super().__init__(predict_fct, do_print=do_print)
        self.n_coalitions, self.n_approx = n_coalitions, n_approx
        self.background_type = kwargs.get("background_type","random")
        self.n_intervals, self.do_add_delta, self.features_map = kwargs.get("n_intervals",10), kwargs.get("do_add_delta",False), kwargs.get("features_map") #for conditional
        self.relevant_params = {"background_type":self.background_type, "n_coalitions":n_coalitions}

    def fit(self, background, background_dict, **kwargs):
        self.expected_value = background_dict["background_expectation"]

        if self.background_type == "conditional":
            self.relevant_params["n_intervals"] = self.n_intervals
            self.relevant_params["do_add_delta"] = self.do_add_delta
        if self.background_type != "baseline":
            self.relevant_params["n_approx"] = self.n_approx
        if self.do_print: self.print_params(self.relevant_params)
            
        if self.background_type == "baseline":
            self.reference_vector = data.get_median(background)
            baseline_prediction_all = self.predict_fct(pd.DataFrame(self.reference_vector,columns=background.columns),return_all=True)
            self.spline_means = baseline_prediction_all.drop(columns=["EstimatedLoad"]).values[0]
            self.expected_value = baseline_prediction_all["EstimatedLoad"].values[0]
        elif self.background_type == "conditional":
            self.background = background #spline means will be computed conditionally to X when called
            self.relevant_params["intervals"] = conditional.get_intervals(background, self.n_intervals, self.features_map)
            self.relevant_params["features_map"] = self.features_map
        
        self.spline_means = background_dict["background_spline_means"]
        assert(self.spline_means is not None)
            
        
    def __call__(self, X, X_dict={}, **kwargs):
        assert(X_dict.get("predictions_all") is not None)
        if self.background_type == "conditional":
            self.spline_means = gams.get_approximate_conditional(self.predict_fct, X, self.background, **self.relevant_params)
        shap_values = X_dict["spline_contribs"] - self.spline_means
        return shap_values
    
    
class SplineExplainer(DefaultExplainer):
    """exact explainer for monovariate GAM"""
    def __init__(self, predict_fct, do_print=False, n_coalitions=10, n_approx=1, **kwargs):
        super().__init__(predict_fct, do_print=do_print)

    def fit(self, background, background_dict, **kwargs):
        self.expected_value = background_dict["background_expectation"]        
        self.spline_means = background_dict["background_spline_means"]
        assert(self.spline_means is not None)
            
    def __call__(self, X, X_dict={}, **kwargs):
        assert(X_dict.get("predictions_all") is not None)
        shap_values = X_dict["spline_contribs"]
        return shap_values

    
class MonteCarloExplainer(DefaultExplainer):
    """Computes Shapley values using Monte Carlo approximation"""
    def __init__(self, predict_fct, do_print=False, n_coalitions=10, n_approx=1, constrain=False, do_reuse=False, **kwargs):
        super().__init__(predict_fct, do_print=do_print)
        
        self.n_coalitions, self.n_approx = n_coalitions, n_approx
        self.constrain, self.do_reuse = constrain, do_reuse
        self.background_type, self.do_double = kwargs.get("background_type", "random"), kwargs.get("do_double",False)
        self.n_intervals, self.features_map = kwargs.get("n_intervals",10), kwargs.get("features_map") #for conditional
        self.aggregation = kwargs.get("aggregation",0)
        self.relevant_params = {"background_type":self.background_type, "n_coalitions":n_coalitions, "constrain":constrain, "do_reuse":do_reuse, "aggregation":self.aggregation}
        
    def fit(self, background, background_dict, **kwargs):
        self.expected_value = background_dict["background_expectation"]
        self.background = background
        if self.background_type == "baseline":
            self.reference_vector = data.get_median(background)
            baseline_prediction = self.predict_fct(pd.DataFrame(np.array(self.reference_vector).reshape(1,background.shape[1]),columns=background.columns))
            self.expected_value = baseline_prediction.values[0]
        else:
            self.background = background
            self.relevant_params["do_double"] = self.do_double
        if self.background_type == "conditional":
            self.relevant_params["n_intervals"] = self.n_intervals
        if self.background_type != "baseline":
            self.relevant_params["n_approx"] = self.n_approx
        if self.do_print: self.print_params(self.relevant_params)
        
        if self.background_type=="conditional":
            self.intervals = conditional.get_intervals(background, self.n_intervals, self.features_map)
            self.relevant_params["intervals"] = self.intervals
            self.relevant_params["features_map"] = self.features_map
        elif self.background_type=="baseline":
            self.relevant_params["baseline"] = self.reference_vector
        if self.constrain == "sampled" or self.constrain == "fixed":
            self.relevant_params["background_expectation"] = self.expected_value
        
    def __call__(self, X, X_dict={}, **kwargs):    
        if self.constrain == "sampled" or self.constrain == "fixed":
            self.relevant_params["predictions"] = X_dict["predictions"]
        outputs = monte_carlo.get_prediction_dataset(X, self.background, do_print=self.do_print, **self.relevant_params)
        if self.do_print:
            print("Total input size to model: ",outputs["prediction_dataset"].shape)
            t1 = perf_counter()
        outputs["coalitions_preds"] = self.predict_fct(outputs["prediction_dataset"]).values
        if self.do_print:
            t2 = perf_counter()
            print(f"Predictions done in {(t2-t1)/60:.3f} min")
        shap_values = monte_carlo.compute_shapley(X, outputs, do_print=self.do_print, **self.relevant_params)
        return shap_values
      

class Shapley:
    """Model to create explainer and compute shapley values"""
    def __init__(self, model, do_print=False, algo="monte_carlo", col_mapping=None, **kwargs):
        
        self.do_print = do_print
        self.algo = algo
        
        self.col_mapping = col_mapping
        self.aggregation = kwargs.get("aggregation",0)
        if self.aggregation != 0 and self.col_mapping is None:
            raise ValueError("Please provide col_mapping for aggregation")
        
        self.background_type = kwargs.get("background_type", "random")
        if self.background_type == "conditional":
            if kwargs.get("features_map") is None:
                raise ValueError("Please provide features_map for conditional") #coder détecter automatiquement feature_map?
            
        if self.do_print:
            print(f"==========Shapley Explainer {self.algo}==========")
            
        #check if model is type MyModel
        if not hasattr(model, "MyModelMarker"):
            self.model = models.MyModel(model)
        else:
            self.model = model
        self.predict_fct = self.model.predict
        self.model_class = self.model.model_class
        self.shap_features = self.model.features

        #explainer
        if self.algo == "splines":
            if self.model_class != "gam":
                raise TypeError("Spline contributions can only be computed on GAM models")
            self.explainer = SplineExplainer(self.model.predict, do_print=do_print, **kwargs)
        elif self.algo=="exact":
            if self.model_class == "gam":
                self.explainer = ExactExplainerGAM(self.model.predict, do_print=do_print, **kwargs)
            else:
                self.explainer = MonteCarloExplainer(self.model.predict, do_print=do_print, n_coalitions=0, n_approx=0, constrain=False, reuse=False, double_sample=True) #very slow
            if self.aggregation != 0:
                print("Exact shap not yet implemented for aggregation") #à ajouter ?
        elif self.algo=="permutation":
            self.explainer = PermutationExplainer(self.model.predict, do_print=do_print, **kwargs)
        elif self.algo=="kernel":
            self.explainer = KernelExplainer(self.model.predict, do_print=do_print, **kwargs)
        elif self.algo=="monte_carlo":
            self.explainer = MonteCarloExplainer(self.model.predict, do_print=do_print, **kwargs)
        else:
            raise ValueError("Please provide a supported algorithm")
        self.fitted = False
        self.called = False
    
    def get_id_col_mapping(self, features):
        """returns col_mapping with col indexes"""
        if self.col_mapping is None:
            return None
        id_col_mapping = {}
        for col in self.col_mapping.keys():
            id_col_mapping[col] = []
            for j, feature in enumerate(features):
                if feature in self.col_mapping[col]:
                    id_col_mapping[col].append(j)
        return id_col_mapping
    
    def pad_cols(self, df):
        """adds 0 to missing columns of df"""
        assert(self.fitted)
        if self.aggregation == 0 and list(df.columns) != self.complete_features:
            for col in self.complete_features:
                if col not in df.columns:
                    df[col] = [0 for k in range(df.shape[0])]
            df = df[self.complete_features]
        return df
    
    def aggregate_cols(self, df, aggregation=0, col_mapping=None):
        """aggregate cols according to col_mapping"""
        if col_mapping is None:
            col_mapping = self.col_mapping
        if col_mapping is not None and aggregation==0:
            merged_shap_values = series.aggregate_series(df, col_mapping)
        else:
            merged_shap_values = df
        return merged_shap_values
    
    def get_contributions(self, df, mean_percents=None):
        """computes contributions from aggregated values"""        
        if mean_percents is None:
            mean_percents = self.mean_percents
        if mean_percents is None:
            contributions = df
            efficiency = self.expected_value + contributions.sum(axis=1)
        else:
            expected_means = mean_percents * self.expected_value #contribution parts of expectation
            contributions = df + expected_means
            efficiency = contributions.sum(axis=1)
        return contributions, efficiency
    
    def get_heightened(self, df, df_mins=None, do_all=True, alpha=1):
        """heightens contributions"""
        if df_mins is None:
            df_mins = self.shap_mins
        delta_df, df_mins, min_sum = series.get_deltas(df, df_mins=df_mins, do_all=do_all, alpha=alpha)
        return delta_df, df_mins, min_sum
        
    def get_errors(self, predictions=None, efficiency=None, percents=None, true_percents=None, spline_percents=None, exact_percents=None, key="C", **kwargs):
        """returns all the errors in percents"""
        
        errors_dict = {}
        if predictions is not None and efficiency is not None:
            error_efficiency = series.score(predictions, efficiency, relative=True, percents=True)
            error_efficiency_abs = series.score(predictions, efficiency, relative=True, percents=True, absolute=True)    
            errors_dict["error_efficiency"] = error_efficiency
            errors_dict["error_efficiency_abs"] = error_efficiency_abs
        
        if percents is not None:
            if true_percents is not None:
                error_smach = series.score(percents[key], true_percents[key], percents=True)
                error_smach_abs = series.score(percents[key], true_percents[key], percents=True, absolute=True)
                errors_dict["error_smach"] = error_smach
                errors_dict["error_smach_abs"] = error_smach_abs

            if spline_percents is not None:
                error_spline = series.score(percents[key], spline_percents[key], percents=True)
                error_spline_abs = series.score(percents[key], spline_percents[key], percents=True, absolute=True)
                errors_dict["error_spline"] = error_spline
                errors_dict["error_spline_abs"] = error_spline_abs

            if exact_percents is not None:
                error_exact = series.score(percents[key], exact_percents[key], percents=True)
                error_exact_abs = series.score(percents[key], exact_percents[key], percents=True, absolute=True)
                errors_dict["error_exact"] = error_exact
                errors_dict["error_exact_abs"] = error_exact_abs
            
        return errors_dict
    
    def get_simple_errors(self):
        """returns rounded absolute errors"""
        if not self.called:
            raise ValueError("Shapley module not yet called on X")
        new_dict = {}
        for key in self.errors_dict.keys():
            if "abs" in key:
                new_dict[key]= round(self.errors_dict[key],2)
        return new_dict
    
    def print_shap_errors(self):
        """prints absolute errors"""
        if not self.called:
            raise ValueError("Shapley module not yet called on X")
        errors_dict = self.get_simple_errors()
        if errors_dict.get("error_efficiency_abs") is not None:
            print(f"efficiency error: {errors_dict['error_efficiency_abs']} %")
        if errors_dict.get("error_smach_abs") is not None:
            print(f"SMACH error: {errors_dict['error_smach_abs']} %")
        if errors_dict.get("error_spline_abs") is not None:
            print(f"GAM error: {errors_dict['error_spline_abs']} %")
        if errors_dict.get("error_exact_abs") is not None:
            print(f"Exact shap error: {errors_dict['error_exact_abs']} %")
    
    def plot_all(self):
        if not self.called:
            raise ValueError("Shapley module not yet called on X")
        plots.show_predictions(self.predictions, self.efficiency, labels=["Predictions","Efficiency"],title=f"Efficiency of {self.algo} Shapley")
        plots.show_sum(self.delta_df, offset=self.expected_value + self.min_sum, title=f"Heightened {self.algo} Shapley contributions")
        plots.show_sum(self.percents,title=f"{self.algo} Shapley percents")
        plots.show_sum(self.parts, title=f"{self.algo} Shapley parts")
    
    def fit(self, background, background_dict={}, **kwargs):
                
        #features
        if isinstance(background, pd.DataFrame):
            self.complete_features = list(background.columns)
            if self.shap_features is None:
                self.shap_features = self.complete_features
        else:
            self.complete_features = self.shap_features
            if self.complete_features is None:
                raise TypeError("Please provide a dataframe or array and features")
        self.id_col_mapping = self.get_id_col_mapping(self.complete_features)
        
        #background
        self.background = data.wrap_data(background, self.shap_features, kwargs.get("y_name","Load"), kwargs.get("index"))
        if self.do_print:
            print(f"Processing background of shape {self.background.shape} - ",end=" ")
            t1 = perf_counter()
        
        self.background_dict = preprocess.get_background_data(self.model, self.background, col_mapping=self.col_mapping, **background_dict, **kwargs)
        if self.do_print:
            t2 = perf_counter()
            print(f"({(t2-t1):.3f} s)")
        
        #mean percents
        self.mean_percents = kwargs.get("mean_percents")
        if self.mean_percents is None:
            self.mean_percents = self.background_dict.get("background_true_mean_percents", None)
        if self.mean_percents is None:
            self.mean_percents = self.background_dict.get("background_spline_mean_percents")
            
        #fit explainer
        if self.do_print:
            print(f"Fitting explainer with params:",end=" ")
            t1 = perf_counter()
        self.explainer.fit(self.background, self.background_dict, **kwargs)
        self.expected_value = self.explainer.expected_value
        self.fitted = True
        if self.do_print:
            t2 = perf_counter()
            print(f"({(t2-t1):.3f} s)")
        
        #shap on background to get shap_mins for heightening
        if kwargs.get("do_background_shap",False):
            if self.do_print:
                print("Computing background shapley values - ",end=" ")
                t1 = perf_counter()
            background_shap_values = self.explainer(self.background, **kwargs)
            background_shap_values = self.pad_cols(background_shap_values)
            background_shap_values = self.aggregate_cols(background_shap_values, self.aggregation)
            background_shap_values, _ = self.get_contributions(background_shap_values)
            _, self.shap_mins, _ = series.get_deltas(background_shap_values, do_all=kwargs.get("do_all",True), alpha=kwargs.get("alpha",1))
            if self.do_print:
                t2 = perf_counter()
                print(f"({(t2-t1):.3f} s)")
        else:
            self.shap_mins = None
        
        

    def __call__(self, X, X_dict={}, **kwargs):
        
        if not self.fitted:
            raise ValueError('Please fit your Shapley module on background data before calling it')
        if self.do_print:
            print("======Computing explanation======")
            t1 = perf_counter()
            
        #X predictions
        self.X = data.wrap_data(X, self.shap_features, kwargs.get("y_name","Load"), kwargs.get("index"))
        self.X_dict = preprocess.get_X_data(self.model, self.X, self.background, background_dict=self.background_dict, col_mapping=self.col_mapping, **X_dict, **kwargs)

        #shap values
        self.shap_values = self.explainer(self.X, self.X_dict, **kwargs)
        self.shap_values = self.pad_cols(self.shap_values)
        self.merged_shap_values = self.aggregate_cols(self.shap_values, self.aggregation)
        self.contributions, self.efficiency = self.get_contributions(self.merged_shap_values, self.mean_percents)
        self.delta_df, self.df_mins, self.min_sum =  self.get_heightened(self.contributions, self.shap_mins, kwargs.get("do_all",True), kwargs.get("alpha",1))
        self.percents = series.get_percents(self.delta_df)
        if self.do_print:
            t2 = perf_counter()
            print(f"({(t2-t1)/60:.3f} min)")
        
        self.predictions = self.X_dict["predictions"]
        self.parts = self.percents.multiply(self.predictions, axis=0)
        self.called = True

        #plots
        if self.do_print:
            print("==========Plots==========")
            self.plot_all()

        #errors
        self.errors_dict = self.get_errors(efficiency=self.efficiency, percents=self.percents, **self.X_dict, key=next(iter(self.col_mapping.keys())))
        if self.do_print:
            print("==========Errors==========")
            self.print_shap_errors()
            
        
        return self.parts
    