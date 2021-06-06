import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Sklearn Transformer to allow selection of columns from a DataFrame

    """

    def __init__(self, columns=[], exclude=[]):
        self.columns = columns
        self.exclude = exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            if len(self.columns):
                X_s =  X[self.columns]
            if len(self.exclude):
                X_s = X[list(set(X.columns).difference(set(self.exclude)))]
                
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
            
        return X_s

        
class PandasTfIdf(BaseEstimator, TransformerMixin):
    
    def __init__(self, text_column, max_features=1000):
        self.max_features = max_features
        self.text_column = text_column
        self.fitted_tf_idf=None

    def fit(self, X, y=None):
        self.fitted_tf_idf = TfidfVectorizer(max_features=self.max_features).fit(X[self.text_column].values)
        return self
    
    def transform(self, X):
        tf_idf = self.fitted_tf_idf.transform(X[self.text_column].values)
        return tf_idf
    
    
def pre_process_ads(ads_df,
                    num_test_ads,
                    max_features=5000,
                    use_location=False):
    """
    
    :param ads_df:
    :param num_test_ads:
    :param max_features:
    :param use_location:
    :return:
    """
    
    # Drop missing values outside of the pre-processing pipeline
    ads_df.dropna(inplace=True)

    target = ads_df['CategoryID'].values

    # Replace category ids with numbers from 1 to num_classes, based on category frequency
    # (1 is most frequent)
    cat_counts = ads_df['CategoryID'].value_counts()
    mapping = dict(zip(cat_counts.index, range(len(ads_df['CategoryID'].unique()))))
    ads_df['CategoryID'] = ads_df['CategoryID'].map(mapping)

    # As per the text, I am assuming that all the classes are represented in the first 100k rows
    num_classes = len(ads_df['CategoryID'].iloc[0:num_test_ads].unique())
    print("Number of classes in the dataset: {}".format(num_classes))

    if not use_location:
        pp_pipeline = Pipeline(steps=[
            ('select_columns', ColumnSelector(['Title', 'CategoryID']))
            , ('tf-idf_from_df', PandasTfIdf('Title', max_features=max_features))
        ])
    else:
        pp_pipeline = Pipeline(steps=[('union', FeatureUnion(
            transformer_list=[
                ('Title_stuff', Pipeline(steps=[('select_title', ColumnSelector(['Title'])),
                                                ('tf_idf',
                                                 PandasTfIdf('Title', max_features=max_features))
                                                ]
                                         )
                 ),
                ('RegionId', ColumnSelector(['RegionID']))
        
            ]
        ))])

    pp_pipeline = pp_pipeline.fit(ads_df)
    X = pp_pipeline.transform(ads_df)
    
    return X, target, pp_pipeline
