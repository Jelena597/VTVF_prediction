import pandas as pd
from random import sample
import sklearn
import seaborn as sb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import quandl
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeCV

def prebaciStringUFloat(columnName):
    data[columnName] = data[columnName].replace(['da'], 1)
    data[columnName] = data[columnName].replace(['ne'], 0)
    data[columnName] = data[columnName].replace(['/'], 2)
    data[columnName] = data[columnName].replace(['muski'], 0)
    data[columnName] = data[columnName].replace(['zenski'], 1)
    data[columnName] = data[columnName].replace(['paroksizmalna'], 1)
    data[columnName] = data[columnName].replace(['permanentna'], 1)
    data[columnName] = data[columnName].replace(['I'], 1)
    data[columnName] = data[columnName].replace(['II'], 2)
    data[columnName] = data[columnName].replace(['III'], 3)
    data[columnName] = data[columnName].replace(['duze od 2 godine'], 10)
    data[columnName] = data[columnName].replace(['krace od 2 godine'], 1)
    data[columnName] = data[columnName].replace(['naprasna srcana'], 1)
    data[columnName] = data[columnName].replace(['nenaprasna srcana'], 2)
    data[columnName] = data[columnName].replace(['naprasna nesrcana'], 3)
    data[columnName] = data[columnName].replace(['nenaprasna nesrcana'], 4)
    data[columnName] = data[columnName].replace([''], 0)


def objasni(data, pca_vr):
    plt.ylim(top=1.1)

    plt.bar(data_pca.columns, pca_vr,
            label='Procenat varijanse')

    plt.plot(data_pca.columns, np.cumsum(pca_vr),
             color='tab:orange',
             label='Kumulativna varijansa',
             marker='x')

    plt.xlabel('Glavne komponente')
    plt.ylabel('Udeo objašnjene varijanse')

    plt.legend()

    plt.show()


def ucitaj(data):
    X = data.iloc[:, 1:-1]
    y = data.loc[:, ['VTVF']]
    vtvf_pojava = data.iloc[:, -1]
    return X, y, vtvf_pojava

data = pd.read_csv("ICD_podaci.csv", sep=';')
#data["Uzrok_smrti"].fillna(0, inplace=True)
#data["vreme_do_prve_VT"].fillna(0, inplace=True)
#data["vreme_do_prve_ICDth"].fillna(0, inplace=True)
pd.set_option('display.max_columns', None)

prebaciStringUFloat('VTVF')
prebaciStringUFloat('VTVF_1')
prebaciStringUFloat('VTVF_2')
prebaciStringUFloat('VTVF_3')
prebaciStringUFloat('prezivljavanje')
prebaciStringUFloat('prezivljavanje_1')
prebaciStringUFloat('prezivljavanje_2')
prebaciStringUFloat('prezivljavanje_3')
prebaciStringUFloat('ICDterapija')
prebaciStringUFloat('ICDterapija_1')
prebaciStringUFloat('ICDterapija_2')
prebaciStringUFloat('ICDterapija_3')
prebaciStringUFloat('IBS')
prebaciStringUFloat('znacajan_broj_VES')
prebaciStringUFloat('AF_pre_ugradnje')
prebaciStringUFloat('HTA')
prebaciStringUFloat('DM')
prebaciStringUFloat('HLP')
prebaciStringUFloat('HOBP')
prebaciStringUFloat('Pusenje')
prebaciStringUFloat('Amiodaron')
prebaciStringUFloat('pol')
prebaciStringUFloat('AF_pre_ugradnje')
prebaciStringUFloat('NYHA')
prebaciStringUFloat('Trajanje_bolesti')
prebaciStringUFloat('Uzrok_smrti')


#OBRADA I IZBACIVANJE
pd.set_option('display.max_columns', None)


y = data.loc[:, ['VTVF']]
X1 = data.iloc[:, 1:25]
X2 = data.iloc[:, 26:]
X = X1.join(X2)
X["vreme_do_prve_VT"].fillna(-100, inplace=True)
X["vreme_do_prve_ICDth"].fillna(-100, inplace=True)
X = X.drop(columns='Uzrok_smrti')
X = X.drop(columns='VTVF_1')
X = X.drop(columns='VTVF_2')
X = X.drop(columns='VTVF_3')
X = X.drop(columns='ICDterapija_1')
X = X.drop(columns='ICDterapija_2')
X = X.drop(columns='ICDterapija_3')
X = X.drop(columns='prezivljavanje')
X = X.drop(columns='prezivljavanje_1')
X = X.drop(columns='prezivljavanje_2')
X = X.drop(columns='prezivljavanje_3')
X = X.drop(columns='SD2')
X = X.drop(columns='SD1')
X = X.drop(columns='vreme_do_prve_VT')
X = X.drop(columns='C1')
X = X.drop(columns='a1resp')
X = X.drop(columns='a2rr')



'''
#KORELACIJA ATRIBUTA
ceo = X.join(y)
print(ceo.shape)
corr = ceo.corr()
dataplot = sb.heatmap(corr, cmap="YlGnBu")
plt.show()
#print(corr)
print(corr.iloc[:, -1])
#corr.to_csv('C:\\Users\\jjelena\\Desktop\\ICD_podaci\\Korelacija.csv', float_format='%.3f')
'''


#SAMO PRVI DEO IZBACEN
'''
ceo = X.join(y)
#ceo.to_csv('C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_prvi.csv')
print(ceo.shape)
sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
X_reduced = sel.fit_transform(X)
print(X.shape)
print(X_reduced.shape)
#print(X[X.columns[sel.get_support(indices=True)]])
print(ceo.columns)
print(X[X.columns[sel.get_support(indices=True)]].columns)
X[X.columns[sel.get_support(indices=True)]].join(y).to_csv('C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_prvi_disp.csv')
'''



#I DRUGI DEO IZBACEN
'''
X = X.drop(columns='C2')
X = X.drop(columns='a2resp')
X = X.drop(columns='a1rr')
ceo = X.join(y)
#ceo.to_csv('C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi.csv')
print(ceo.shape)
sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
X_reduced = sel.fit_transform(X)
print(X.shape)
print(X_reduced.shape)
#print(X[X.columns[sel.get_support(indices=True)]])
print(ceo.columns)
print(X[X.columns[sel.get_support(indices=True)]].columns)
X[X.columns[sel.get_support(indices=True)]].join(y).to_csv('C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi_disp.csv')
'''


'''
#I BEZ DUZINE PRACENJA
X = X.drop(columns='C2')
X = X.drop(columns='a2resp')
X = X.drop(columns='a1rr')
X = X.drop(columns='duzina_pracenja')
ceo = X.join(y)
#ceo.to_csv('C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno.csv')
print(ceo.shape)
sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
X_reduced = sel.fit_transform(X)
print(X.shape)
print(X_reduced.shape)
#print(X[X.columns[sel.get_support(indices=True)]])
print(ceo.columns)
print(X[X.columns[sel.get_support(indices=True)]].columns)
X[X.columns[sel.get_support(indices=True)]].join(y).to_csv('C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno_disp.csv')
'''


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


X, y, vtvf_pojava = ucitaj(data10)


ceo = X.join(y)
vtvf_pojava = vtvf_pojava.map(lambda x: bool(x))
ceo.iloc[:, -1] = vtvf_pojava
print("ceo")
print(ceo)

'''
vrednost,kolicina = np.unique(vtvf_pojava, return_counts=True)
suma = np.sum(kolicina)
print("suma")
print(suma)


#Vidimo da je bilo vise onih kod kojih nije doslo do pojave vtvf stanja nego sto je bilo
labels = 'Nije se ispoljio VTVF', 'Ispoljen VTVF'
sizes = kolicina
fig, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels, autopct='%1.3f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()
'''

#korelacija atributa
korelacija = ceo.corr()
ciljni_korelacija = korelacija.iloc[:, -1]
ciljni_korelacija.sort_values(ascending=False, key=abs)
print("ciljni korelacija")
print(ciljni_korelacija)

#probamo normalizaciju i proveravamo disperziju
norm = normalize(ceo, axis=0)
normalizovanCeo = pd.DataFrame(norm, ceo.index, ceo.columns)

var = np.var(normalizovanCeo, axis=0)
var.sort_values(ascending=False)

print("var")
print(var)


#probamo PCA
podaci = ceo.iloc[:, :-1]

pca = PCA()

data_pca = pd.DataFrame(pca.fit_transform(podaci),
                      index = ceo.index,
                      columns = [f'pca{i}' for i in range(1, ceo.shape[1])])
print("data_pca")
print(data_pca)

pca_vr = pd.Series(pca.explained_variance_ratio_,
                index = data_pca.columns)
print("pca_vr")
print(pca_vr)

data_pca = data_pca.iloc[:, :3]
pca_vr = pca_vr[:3]
objasni(data_pca, pca_vr)

_, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 2))

for i in range(3):
    ax = axs[i]

    if i != 0:
        ax.scatter(data_pca[~vtvf_pojava].iloc[:, 0],
                   data_pca[~vtvf_pojava].iloc[:, 1],
                   color='tab:blue', alpha=.8,
                   label='Nema vtvf')

    if i != 1:
        ax.scatter(data_pca[vtvf_pojava].iloc[:, 0],
                   data_pca[vtvf_pojava].iloc[:, 1],
                   color='tab:orange', alpha=.8,
                   label='Ima vtvf')

plt.legend()
plt.tight_layout()

plt.show()

ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_[0])
print(ridge.coef_[0])
feature_names = np.array(X.columns)
plt.bar(height=importance, x=feature_names)
plt.xticks(rotation='vertical')
plt.title("Znacajnost atributa")
plt.show()


'''
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, np.ravel(y,order='C'))
ranking = rfe.ranking_.reshape(7, 7)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()



svc = SVC(kernel="linear")
# The "accuracy" scoring shows the proportion of correct classifications

min_features_to_select = 7  # Minimum number of features to consider
rfecv = RFECV(
    estimator=svc,
    step=1,
    cv=StratifiedKFold(3),
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
)
rfecv.fit(X, np.ravel(y,order='C'))

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
    rfecv.grid_scores_,
)
plt.show()

lsvc = LinearSVC(C=0.07, penalty="l2", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_reduced = model.transform(X)
print('X_reduced')
print(X_reduced.shape)
for i in range(-3, 3):
    print(2**i)
'''



