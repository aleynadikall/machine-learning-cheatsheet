import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataset/multiple-linear-regression-dataset.csv')
deneyim_yas = data.loc[:, ['deneyim', 'yas']].values
maas = data['maas'].values.reshape(-1, 1)


import sklearn.linear_model as lm

reg = lm.LinearRegression()

import sklearn.model_selection as ms

x_train, x_test, y_train, y_test = ms.train_test_split(deneyim_yas, maas, test_size=1 / 3, random_state=0)

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print('Deneyim ve Yaşlar: ', x_test)
print("Tahmin Edilen Maaşlar: ", y_pred)

#score
import sklearn.metrics as mt
score = mt.r2_score(y_test, y_pred)
print('score:', score)

#graph
plt.scatter(deneyim_yas[:, 0], maas, color="r")
plt.scatter(x_test[:, 0], y_pred, color='b')
plt.show()

"""
plt.scatter(deneyim_yas[:, 1], maas, color="r")
plt.scatter(x_test[:, 1], y_pred, color='b')
plt.show()
"""