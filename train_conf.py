# This file contains configuration parameters
# for training and pre-processing
# It will be executed using runpy and all the variables contained here
# will be available via a dictionary

num_total_ads = 5000  # Total number of ads to be considerd
num_test_ads = 1000  # Number of ads to be used as test set (set is selected from 0, not random)
use_location = True  # Whether or not to use Location dataset features
location_features = ['RegionID']  # List of location features to use
random_state = 42  # Random seed to ensure repeatability

train_params = {'objective': 'multi:soft_prob',
                'n_estimators': 25,
                'early_stopping_rounds': 5}
