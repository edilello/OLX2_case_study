import joblib
import logging
import pandas as pd
import runpy


def run_inference(conf_file='inference_conf.py'):
    """
    Predict the category of an ad using the ad title and, depending on the how the model was
    trained, location info. Ads come from the data/AdsInfo.tsv or must be contained in a similarly
    formatted pandas DataFrame
    
    :param conf_file: Python file containing the inference configuration parameters
    :return: np.array
             the k most likely CategoryID for the ad
    """
    logger = logging.getLogger(__name__)
    logger.info("Running inference...")
    
    
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_inference()
