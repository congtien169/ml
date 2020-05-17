import pandas as pd
from joblib import load


class TextClassificationPredict(object):
    def __init__(self):
        self.test = None

    def get_train_data(self):

        test_data = []
        test_data.append({"feature": u"Trưởng ban Dân nguyện Nguyễn Thanh Hải sẽ làm Bí thư tỉnh ủy Thái Nguyên"})
        df_test = pd.DataFrame(test_data)

        clf = load('filename.joblib')
        predicted = clf.predict(df_test["feature"])
        # Print predicted result
        print(predicted)
        result_score = clf.predict_proba(df_test["feature"])
        print(result_score)


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()