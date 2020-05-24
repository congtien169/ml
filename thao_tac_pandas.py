import pandas as pd
from sklearn.metrics import classification_report
from NaiveBayesModel import NaiveBayesModel
from svm_pp import SVMModel


class TextClassificationPredict(object):
    def __init__(self):
        self.test = None

    def get_train_data(self):
        df_train = pd.read_csv('test.csv', index_col=0)
        # create model
        model = SVMModel()
        # traning model
        # data test
        test_data = []
        test_data.append({"feature": u"Huế là thành phố áo dài, tại sao không",
                          })
        df_test = pd.DataFrame(test_data)

        result_test = []
        clfx = model.clf.fit(df_train.feature, df_train.target)

        for i in range(100):
            predicted = clfx.predict(df_test.feature)
            y_train_pred = classification_report(df_train.target, model.clf.predict(df_train.target), output_dict=True,
                                             zero_division=1)
            ds = clfx.predict_proba(df_test["feature"])
            count = 0
            for i in y_train_pred:
                if i == predicted[0]:
                    diemso = ds[0][count]
                    break
                count += 1
            print('label: {0} - diem so: {1}'.format(predicted[0], diemso))
            #print(y_train_pred)
            result_test.append({'predicted':predicted[0],'diemso':diemso})
            #print(y_train_pred[predicted[0]])
            #print(ds)


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()
