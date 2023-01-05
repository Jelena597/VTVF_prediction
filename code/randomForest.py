import pandas as pd
from random import sample
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_selection import VarianceThreshold
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


X, y = ucitaj(data10)
'''
y = data.loc[:, ['VTVF']]
print(y)
y = pd.factorize(data['VTVF'])[0]

y = y.transpose()
y_pred = np.reshape(y_predicted, (-1, 1))
y_pred = y_pred.transpose()
'''

'''
data["Uzrok_smrti"].fillna(0, inplace=True)
data["vreme_do_prve_VT"].fillna(0, inplace=True)
data["vreme_do_prve_ICDth"].fillna(0, inplace=True)


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
              'criterion': ['gini'],
              'n_estimators': [100, 200, 300, 500, 1000],
              'max_features': ['sqrt', 'log2']
          },
          {
              'criterion': ['entropy'],
              'n_estimators': [100, 200, 300, 500, 1000],
              'max_features': ['sqrt', 'log2']
          },
          {
              'criterion': ['log_loss'],
              'n_estimators': [100, 200, 300, 500, 1000],
              'max_features': ['sqrt', 'log2']
          }
         ]

rf = RandomForestClassifier()
model = GridSearchCV(rf,param_grid=params, cv=5, scoring=['balanced_accuracy','roc_auc'], verbose=4, refit='balanced_accuracy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
model.fit(X_train, np.ravel(y_train, order='C'))

print('Best estimator')
print(model.best_estimator_)
print(model.best_params_)

y_predicted = model.best_estimator_.predict(X_test)


confM = confusion_matrix(y_test, y_predicted)
print(confM)

accScore = accuracy_score(y_test, y_predicted)
print(accScore)

#modelReduced = SelectFromModel(clf, prefit=True)
#X_reduced = modelReduced.transform(X)
#print(X_reduced.shape)

prikazi_rezultate(model, X_test, y_test, y_predicted)
