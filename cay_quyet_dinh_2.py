from sklearn import tree
import numpy as np
my_tree = tree.DecisionTreeClassifier()

feature_chu = [
    ['nhe', 'tb', 'tb', 'nhieu'],
    ['nang', 'thap', 'cao', 'it'],
    ['nhe', 'thap', 'cao', 'it'],
    ['nang', 'cao', 'cao', 'tb'],
    ['nhe', 'cao', 'cao', 'nhieu'],
    ['tb', 'thap', 'tb', 'nhieu'],
    ['tb', 'tb', 'tb', 'it'],
    ['nang', 'thap', 'thap', 'nhieu']
]

np_feature_chu = np.array([
    ['nhe', 'tb', 'tb', 'nhieu'],
    ['nang', 'thap', 'cao', 'it'],
    ['nhe', 'thap', 'cao', 'it'],
    ['nang', 'cao', 'cao', 'tb'],
    ['nhe', 'cao', 'cao', 'nhieu'],
    ['tb', 'thap', 'tb', 'nhieu'],
    ['tb', 'tb', 'tb', 'it'],
    ['nang', 'thap', 'thap', 'nhieu']
])


quy_uoc_feature = [
    ['nhe', 'thap', 'tb', 'cao', 'nang', 'it', 'nhieu'],
    [1, 2, 3, 4, 5, 6, 7]
]

# get all value from row 0 and col 0
# print(np_feature_chu[0:,0])

# get all value from row 1 and col 0
#print('get all value from row 1 and col 0')
#print(np_feature_chu[1:, 0])

# get all value from row 0 and col 1
# print(np_feature_chu[0:,1])

# get value of row 0 and col 3 (inc 0)
print(np_feature_chu[0,0])
print(np_feature_chu[1,0])

# get all value from row 0 to row 3
#print(np_feature_chu[0:3])

row, col = np_feature_chu.shape
#print(row, col)
np_feature_so = np.empty_like(np_feature_chu)

for i in range(row):
    for j in range(col):
        np_feature_so[i,j] = (np_feature_chu[i,j])

print(np_feature_so)
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

tagget_chu = ['khong', 'co', 'co', 'khong', 'khong', 'khong', 'khong', 'co']

tagget_so = ['0', '1', '1', '0', '0', '0', '0', '1']

#my_tree.fit(X=feature_so, y=tagget_so)

feature_test_chu_1 = [['nhe', 'cao', 'tb', 'it']]
feature_test_chu_2 = [['nhe', 'cao', 'tb', 'nhieu']]

feature_test_so_1 = [['1', '4', '3', '6']]
feature_test_so_2 = [['1', '4', '4', '7']]

#result = my_tree.predict(X=feature_test_so_2)
#print(type(result))
#print(result)
