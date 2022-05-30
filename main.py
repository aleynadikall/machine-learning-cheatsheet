# Makine öğrenmesi denilen şey aslında cebir geometri ve bilgisayar biliminin bir araya gelmesinden oluşuyor.
"""
Temelde deistatistik çalışmaları yapılıyor.
Belli bir bölgedeki evlerin fiyatlarını belirlemek istediğimizde elimizdeki datayla machine learning çalışmalarıyla bunu
belirleyebiliyoruz.
Herhangi birisinin bir hastalığa sahip olup olmadığını machine learning çalışması ile belirleyebiliyoruz.
Eldeki röntgen verilerinden tanı yapmak istendiğinde machine learningten faydalanılıyor.
Hava durumu için yine machine larning kullanılabiliyor.

"""
"""
Machine learning 3 alt kırılıma ayrılabiliyor.
Gözetimli (supervised) öğrenme: Elimizde bir dataset(Etiketlenmiş) olmalı ve bu datayı train ve test olarak ayırırız. Optimum değerlere
ulaşana kadar modelimizi eğitiriz.
Gözetimsiz (unsupervised) Öğrenme: Etiketlenmemiş datalarla çalışırız. Nasıl sınıflandıracağını biz söylemeyiz makine öğrenmesi ortak 
özelliklerden faydalanarak datalari kümelere ayirir.
Semi-supervised öğrenme

"""

"""
Life cycle:
1-Collecting Data
2-Data Wrangling
3-Analyse Data
4-Train Algorithm
5-Test Algorithm
6-Deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Gözetimli öğrenme:
1-Linear Regression:Maaş tahmini, ev fiyatları tahmini, arabanın fiyatını belirlemek
    -Birden fazla veriyle yapılan regresyon multiple linear regression'dır.
2-Logistic Regression: Amerikadaki seçimlerde Trump mı kazanır Biden mı kazanır kararını verdirmek için kullanılabilir.
3-Polinominal Regression: Hava durumu tahmini
4-Decision Tree Regression: Aynı ağaçlar gibi dallanarak kırılımlar üzerinden ilerleyen bir yapıya sahiptir.
5-Random Forest: İçinde ağaçlar olan rassal orman
"""

data = pd.read_csv('dataset/linear-regression-dataset.csv')

# tüm sütun boyunca dataları aldık.
deneyim = data['deneyim'].values.reshape(-1, 1)
maas = data['maas'].values.reshape(-1, 1) # bu bize tek boyutlu matrisi vermiş olacak

# algorithm
import sklearn.linear_model as lm  # lineer modelimizi çağırdık

reg = lm.LinearRegression()

# data split
import sklearn.model_selection as ms  # datanın train test olarak ikiye bolunebilmesini sağlar.

x_train, x_test, y_train, y_test = ms.train_test_split(deneyim, maas, test_size=1 / 3, random_state=0)

# train (fit etmek)
reg.fit(x_train, y_train)

# predict
y_pred = reg.predict(x_test)

print('Deneyimler:', x_test)
print('Tahmin Edilen Maaslar:', y_pred)

#score
import sklearn.metrics as mt
score = mt.r2_score(y_test,y_pred)
print('score:',score)

#graph
plt.scatter(deneyim, maas, color='r')
plt.scatter(x_test, y_pred, color='b')
plt.show()