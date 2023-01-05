import pandas as pd
from random import sample
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
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


X, y = ucitaj(data5)

X_da = pd.DataFrame()
X_ne = pd.DataFrame()

'''
for index, row in data5.iterrows():
    print(row)
    if row['VTVF'] == 1:
        print('DA')
        X_da = X_da.join(row)
    elif row['VTVF'] == 0:
        X_ne = X_ne.join(row)
'''

X_da = X_da.append(data5[data5['VTVF'] == 1])
X_ne = X_ne.append(data5[data5['VTVF'] == 0])
y_da = X_da.loc[:, ['VTVF']]
y_ne = X_ne.loc[:, ['VTVF']]
X_da = X_da.iloc[:, 1:-1]
X_ne = X_ne.iloc[:, 1:-1]


#X["vreme_do_prve_ICDth"].fillna(0, inplace=True)

#scaler = StandardScaler()
#X = scaler.fit_transform(X_reduced)
#print(X.shape)

#norm = Normalizer()
#X = norm.fit_transform(X)

#quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
#X = quantile_transformer.fit_transform(X)
#X[np.isnan(X)] = -100




#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_da, y_da, test_size=0.3, stratify=y_da)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_ne, y_ne, test_size=0.3, stratify=y_ne)


'''

quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
X_ne = quantile_transformer.fit_transform(X_ne)
X_ne[np.isnan(X_ne)] = -100

quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
X_da = quantile_transformer.fit_transform(X_da)
X_da[np.isnan(X_da)] = -100
'''

params = [{
              'C': [0.01, 0.03, 0.1, 0.125, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.7, 3],
              'kernel': ['poly'],
              'degree' : [3, 5],
              'gamma': np.arange(0.1, 0.5, 1)
          },
          {
              'C': [0.01, 0.03, 0.1, 0.125, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.7, 3],
              'kernel': ['linear']
          },
          {
              'C': [0.01, 0.03, 0.1, 0.125, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.7, 3],
              'kernel': ['rbf'],
              'gamma': np.arange(0.1, 0.5, 1)
          },
{
              'C': [0.01, 0.03, 0.1, 0.125, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.7, 3],
              'kernel': ['sigmoid'],
              'gamma': np.arange(0.1, 0.5, 1)
          }
         ]


model = GridSearchCV(SVC(), param_grid=params, cv=5, scoring=['balanced_accuracy','roc_auc'], verbose=4, refit='balanced_accuracy')
model.fit(X_train, np.ravel(y_train,order='C'))
print('Best estimator')
print(model.best_estimator_)
print('N support')
print(model.best_estimator_.n_support_)
print('Support')
print(model.best_estimator_.support_)

y_predict = model.best_estimator_.predict(X_test)

confM = confusion_matrix(y_test, y_predict)
print(confM)

accScore = accuracy_score(y_test, y_predict)
print(accScore)

prikazi_rezultate(model, X_test, y_test, y_predict)
