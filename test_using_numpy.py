import numpy as np

list_1 = []
list_1.append(0)
list_1.append(1)
list_1.append(2)
list_1.append(3)
list_2 = []
list_2.append(6)
list_2.append(7)
list_2.append(8)
list_2.append(9)
a = []
a.append(list_1)
a.append(list_2)
np_a = np.array(a)

print(np_a)
