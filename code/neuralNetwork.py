import pandas as pd
from random import sample
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BicScore, PC, TreeSearch, ExhaustiveSearch
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

def ucitaj(data):
    X = data.iloc[:, 1:-1]
    y = data.loc[:, ['VTVF']]
    return X, y

def prikazi_rezultate(model, x_test, y_test, y_pred):
    print(classification_report(y_test, y_pred))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    plot_confusion_matrix(model, x_test, y_test, ax=ax1, cmap=plt.cm.Blues, colorbar=False)
    ax1.set_xlabel('Predviđeno')
    ax1.set_ylabel('Stvarno')
    ax1.set_title('Matrica konfuzije')

    plot_roc_curve(model, x_test, y_test, pos_label=False, lw=3, color='orange', ax=ax2)
    ax2.set_xlabel('Stopa lažno neuspešnih')
    ax2.set_ylabel('Stopa stvarno neuspešnih')
    ax2.set_title('ROC kriva neuspešnih')
    ax2.legend(loc='lower right')

    plt.show()

    return model

pd.set_option('display.max_columns', None)
data1 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_prvi.csv", sep=',')
data2 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_prvi_disp.csv", sep=',')
data3 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi.csv", sep=',')
data4 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi_disp.csv", sep=',')
data5 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno.csv", sep=',')
data6 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno_disp.csv", sep=',')
data7 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi_ICD.csv", sep=',')
data8 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi_disp_ICD.csv", sep=',')
data9 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno_ICD.csv", sep=',')
data10 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno_disp_ICD.csv", sep=',')

print(data9.shape)
print(data10.shape)

X, y = ucitaj(data10)

'''
data["Uzrok_smrti"].fillna(-100, inplace=True)
data["vreme_do_prve_VT"].fillna(-100, inplace=True)
data["vreme_do_prve_ICDth"].fillna(-100, inplace=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

#norm = Normalizer()
#X = norm.fit_transform(X)

quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
X = quantile_transformer.fit_transform(X)
X[np.isnan(X)] = -100
print(X)

'''

quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
X = quantile_transformer.fit_transform(X)
X[np.isnan(X)] = -100

params = [{
              'solver': ['lbfgs'],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2],
              'max_iter': [3000],
          },
          {
              'solver': ['sgd'],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2],
              'learning_rate': ['constant', 'invscaling', 'adaptive'],
              'max_iter': [3000]
          },
          {
              'solver': ['adam'],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2],
              'max_iter': [3000]
          }
         ]

model = GridSearchCV(MLPClassifier(), params, cv=5, scoring='accuracy', verbose=4)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
#model = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=600, random_state=1231)
model.fit(X_train, np.ravel(y_train,order='C'))

print('Best estimator')
print(model.best_estimator_)

y_predict = model.best_estimator_.predict(X_test)

confM = confusion_matrix(y_test, y_predict)
print(confM)

accScore = accuracy_score(y_test, y_predict)
print(accScore)

prikazi_rezultate(model, X_test, y_test, y_predict)