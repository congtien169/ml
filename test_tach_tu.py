import pandas as pd
from joblib import dump, load
#from naive_bayes_model import NaiveBayesModel
from sklearn.metrics import classification_report

from svm_pp import SVMModel


class TextClassificationPredict(object):
    def __init__(self):
        self.test = None

    def get_train_data(self):
        # Tạo train data
        train_data = []
        train_data.append({"feature": u"Đà Nẵng sẽ là đô thị một cấp chính quyền", "target": "thoi_su"})
        train_data.append({"feature": u"Bị chém rớt cánh tay giữa phố do mâu thuẫn", "target": "phap_luat"})
        train_data.append({"feature": u"Xây hai cây cầu trọng điểm kết nối Hải Phòng với Hải Dương", "target": "thoi_su"})
        train_data.append({"feature": u"Ninh Thuận công bố hạn hán cấp độ 3", "target": "thoi_su"})
        train_data.append({"feature": u"Lập biên bản tổng giám đốc Bayer Việt Nam về hành vi gửi tài liệu có 'đường lưỡi bò'", "target": "phap_luat"})
        train_data.append({"feature": u"Đến ngôi nhà một hộ cận nghèo 'tự nguyện’ không nhận tiền hỗ trợ", "target": "cong_dong"})
        train_data.append({"feature": u"Uống bia xong tắm biển gặp nạn bơi cứu nhau, 1 người bị sóng cuốn chết đuối", "target": "tai_nan"})
        train_data.append({"feature": u"Uống bia xong tắm biển gặp nạn bơi cứu nhau, 1 người bị sóng cuốn chết đuối","target": "nhau"})
        train_data.append({"feature": u"Thêm 4 bệnh nhân COVID-19 từ nước ngoài về, đều cách ly ngay khi nhập cảnh", "target": "suc_khoe"})
        train_data.append({"feature": u"Thêm 4 bệnh nhân COVID-19 từ nước ngoài về, đều cách ly ngay khi nhập cảnh","target": "y_te"})
        train_data.append({"feature": u"Thứ trưởng Lê Quang Tùng: Làm thế nào để người dân có tiền du lịch?", "target": "thoi_su"})
        train_data.append({"feature": u"Cử tri đề nghị Quốc hội ra nghị quyết về Biển Đông", "target": "chinh_tri"})
        train_data.append({"feature": u"Hai cha con đánh tới tấp bảo vệ, điều dưỡng ngay phòng cấp cứu", "target": "phap_luat"})

        df_train = pd.DataFrame(train_data)
        #print(df_train)
        # Tạo test data
        test_data = []
        test_data.append({"feature": u"Xông vô bệnh viện 'giải vây' bị can rồi quay clip tung tin 'công an đánh người'"})
        df_test = pd.DataFrame(test_data)

        # init model naive bayes
        model = SVMModel() #NaiveBayesModel()

        clf = model.clf.fit(df_train.feature, df_train.target)

        # save to file
        # dump(clf, 'filename.joblib')
        predicted = clf.predict(df_test["feature"])
        # check report about 'label' data train
        y_train_pred = classification_report(df_train.target, model.clf.predict(df_train.target),output_dict=True,zero_division=1)
        print(type(y_train_pred))
        print(y_train_pred.keys())


        # Print predicted result

        print(predicted)
        result_score = clf.predict_proba(df_test["feature"])
        print(result_score)
        print(result_score[0][1])


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()