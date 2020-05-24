import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from NaiveBayesModel import NaiveBayesModel
from svm_pp import SVMModel

import matplotlib.pyplot as plt

class TextClassificationPredict(object):
    def __init__(self):
        self.test = None

    def get_train_data(self):
        df_train = pd.read_csv('test.csv', index_col=0)
        # model = NaiveBayesModel() #SVMModel()

        X_train, X_test, y_train, y_test = train_test_split(df_train["feature"], df_train["target"],
                                                            test_size=0.1,
                                                            random_state=50)

        model = SVMModel()

        # clfx = model.clf.fit(df_train.feature, df_train.target)
        for i in range(100):
            clfx = model.clf.fit(X_train, y_train)

        test_data = []
        test_data.append({"feature": u"Chính phủ yêu cầu đánh giá lại toàn tuyến kè hộ thành hào kinh thành Huế",
                          "target": "thoi_su"})
        df_test = pd.DataFrame(test_data)

        predicted = clfx.predict(df_test.feature)
        sco = clfx.score(X_test, y_test)
        # sco = clfx.score(df_test["feature"], df_test["target"])
        print(sco)
        print(X_test)

        print('======================')
        new_row = pd.Series({"feature": test_data[0]["feature"], "target": predicted[0]})
        df_train = df_train.append(new_row, ignore_index=True)
        # df_train.to_csv('test.csv')
        print('======================')

        y_train_pred = classification_report(df_train.target, model.clf.predict(df_train.target), output_dict=False,
                                             zero_division=1)
        print(y_train_pred)
        ds = clfx.predict_proba(df_test["feature"])
        count = 0
        for i in y_train_pred:
            if i == predicted[0]:
                diemso = ds[0][count]
                break
            count += 1
        print('label: {0} - diem so: {1}'.format(predicted[0], diemso))
        print(y_train_pred[predicted[0]])
        print(ds)


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()
