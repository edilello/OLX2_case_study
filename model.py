import numpy as np


class BestKCatEstimatorWithThreshold:
    
    def __init__(self, pp_pipeline, model, k=3, prob_thresh=0.25):
        self.pp_pipeline = pp_pipeline
        self.model = model
        self.k = k
    
    def predict(self, df, prob_thresh):
        """
        Pre-process the provided DataFrame, compute predictions,
        and return most probable k-classes
        """
        
        # Apply pre-processing pipeline to df
        df_pp = self.pp_pipeline.transform(df)
        
        # Compute class probabilities
        predictions = self.model.predict_proba(df_pp).squeeze()
        
        # Return index if k classes with highest probability
        best_k = np.argsort(predictions)[::-1][0:self.k]
        
        if not (predictions[best_k[self.k - 1]] >= prob_thresh):
            best_k = None
            
        return best_k
