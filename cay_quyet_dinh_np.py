from sklearn import tree
import numpy as np

np_quy_uoc_feature = np.array([
    ['nhe', 'thap', 'tb', 'cao', 'nang', 'it', 'nhieu'],
    [1, 2, 3, 4, 5, 6, 7]
])


def find_number(giatri):
    row, col = np_quy_uoc_feature.shape
    for j in range(col):
        if giatri == np_quy_uoc_feature[0, j]:
            return np_quy_uoc_feature[1, j]
    return None


def assign_number_tagert(giatri):
    if giatri == 'khong':
        return 0
    if giatri == 'co':
        return 1
    return None


my_tree = tree.DecisionTreeClassifier()

np_feature_chu = np.array([
    ['khong', 'nhe', 'tb', 'tb', 'nhieu'],
    ['co', 'nang', 'thap', 'cao', 'it'],
    ['co', 'nhe', 'thap', 'cao', 'it'],
    ['khong', 'nang', 'cao', 'cao', 'tb'],
    ['khong', 'nhe', 'cao', 'cao', 'nhieu'],
    ['khong', 'tb', 'thap', 'tb', 'nhieu'],
    ['khong', 'tb', 'tb', 'tb', 'it'],
    ['co', 'nang', 'thap', 'thap', 'nhieu']
])
# tách feature và target ra khỏi np array
feature_chu = np_feature_chu[0:, 1:5]
print('feature_chu')
print(feature_chu)
target_chu = np_feature_chu[0:, 0]
print('target_chu')
print(target_chu)

feature_so = np.empty_like(feature_chu)
row, col = feature_chu.shape
for i in range(row):
    for j in range(col):
        feature_so[i, j] = find_number(feature_chu[i, j])
print(feature_so)

target_so = np.empty_like(target_chu)
row_target = target_chu.shape
#print(type(row_target))
#print(row_target[0])
for i in range(row_target[0]):
    print(target_chu[i])
    target_so[i] = assign_number_tagert(target_chu[i])
print(target_so)

"""
feature_so = [
    ['1', '3', '3', '7'],
    ['5', '2', '4', '6'],
    ['1', '2', '4', '6'],
    ['5', '4', '4', '6'],
    ['1', '4', '4', '7'],
    ['3', '2', '3', '7'],
    ['3', '3', '3', '6'],
    ['5', '2', '2', '7']
]
tagget_so = ['0', '1', '1', '0', '0', '0', '0', '1']
"""

my_tree.fit(X=feature_so, y=target_so)

"""
np_quy_uoc_feature = np.array([
    ['nhe', 'thap', 'tb', 'cao', 'nang', 'it', 'nhieu'],
    [1, 2, 3, 4, 5, 6, 7]
])
"""
feature_test_chu_1 = [['nhe', 'cao', 'tb', 'it']]
feature_test_chu_2 = [['nhe', 'cao', 'tb', 'nhieu']]

feature_test_so_1 = [['1', '2', '5', '7']]
feature_test_so_2 = [['1', '4', '4', '7']]

result = my_tree.predict(X=feature_test_so_1)
print(type(result))
print(result)

