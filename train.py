import logging
import numpy as np
import pandas as pd
import pre_processing as pp
import utils
import runpy
import xgboost as xgb
from model import BestKCatEstimatorWithThreshold
from sklearn.model_selection import train_test_split


def run_train(conf_file='train_conf.py'):
    """
    Train a model for Category prediction for the OLX2 case study
    
    :param conf_file: Python file containing the training configuration parameters
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Training model")
    
    # Load configuration parameters
    conf = runpy.run_path(conf_file)
    print('loaded configuration file')

    ads_filepath = conf.get('ads_filepath', 'data/AdsInfo.tsv')
    num_total_ads = conf.get('num_total_ads', 5000)
    num_test_ads = conf.get('num_test_ads', 1000)
    use_location = conf.get('use_location', False)
    
    # Ads data
    ads_df = pd.read_csv(ads_filepath, sep='\t', nrows=num_total_ads)

    # if Location data is to be used, merge with ads DataFrame
    if use_location:
        location_filepath = conf.get('location_filepath', 'data/Location.tsv')
        location_df = pd.read_csv(location_filepath, sep='\t')
        ads_df = ads_df.merge(location_df,  on='LocationID')
        
    print(f'Dataset loaded, use_location = {use_location}')
    
    # Pre-process DataFrame, return features, target and fitted pp_pipeline
    X, target, pp_pipeline = pp.pre_process_ads(ads_df,
                                                num_test_ads,
                                                use_location=use_location)
    
    # Save processed feature matrix and target vector
    utils.save_to_disk(X, 'output/', 'features.pkl')
    utils.save_to_disk(X, 'output/', 'target.pkl')

    #  Random split train and validation sets, test set is fixed to the first 100k ads
    X_test = X[0:num_test_ads, :]
    X_train = X[num_test_ads + 1:, :]
    
    y_test = target[0:num_test_ads]
    y_train = target[num_test_ads + 1:]
    
    random_state = conf['random_state']
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1,
                                                          random_state=random_state)

    # Train vanilla Xgboost model with soft_prob objective, no hyperparams optimization for now
    num_classes = np.unique(target)
    train_params = conf['train_params']
    
    clf = xgb.XGBClassifier(**train_params, num_class=num_classes)
    
    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=['mlogloss'],
            verbose=True)
    
    evals_result = clf.evals_result()
    
    
    # Evaluate model on train, valid and test sets
    train_pred_proba = clf.predict_proba(X_train)
    valid_pred_proba = clf.predict_proba(X_valid)
    test_pred_proba = clf.predict_proba(X_test)
    
    # Save Model
    model_with_thresh = BestKCatEstimatorWithThreshold(pp_pipeline, clf, prob_thresh=0.01)
    
    utils.save_to_disk(model_with_thresh, 'models', 'model.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_train()

