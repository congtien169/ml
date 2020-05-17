from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tach_tu import FeatureTransformer
from sklearn.linear_model import SGDClassifier


class SVMModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf-svm", SGDClassifier(
                loss='log',
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                learning_rate='optimal',
                n_jobs=6,
                average=3
            ))
        ])

        return pipe_line