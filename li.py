import pandas as pd
import matplotlib.pyplot as plt
dataframe = pd.read_csv('Advertising.csv')
X = dataframe.values[:, 2]
y = dataframe.values[:, 4]
plt.scatter(X, y, marker='o')
plt.show()
print(dataframe)