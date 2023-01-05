import pandas as pd
from random import sample
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.multiclass import unique_labels
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel


from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearch, BicScore, PC, TreeSearch, ExhaustiveSearch, BDeuScore, K2Score, BDsScore
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, MmhcEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, plot_roc_curve

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

def klasifikuj(model, _test_data, vtvf):
    inference = VariableElimination(model)

    y = []
    y_pred = []

    for index, row in _test_data.iterrows():
        y.append(row[vtvf])
        ev = {}
        for att in vtvf:
            ev[att] = X.loc[0, att]

        y_pred.append(inference.map_query([vtvf], evidence=ev, show_progress=False)[vtvf])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    # plot_confusion_matrix(model, X_test, Y_test, ax=ax1, cmap=plt.cm.Blues, colorbar=False)
    cm = confusion_matrix(y, y_pred)
    display_labels = unique_labels(y, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(include_values=True, cmap=plt.cm.Blues, ax=ax1, xticks_rotation='horizontal', colorbar=False)
    ax1.set_xlabel('Predviđeno')
    ax1.set_ylabel('Stvarno')
    ax1.set_title('Matrica konfuzije')

    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=False)
    roc_auc = auc(fpr, tpr)

    viz = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        # estimator_name=name,
        pos_label=False
    )

    viz.plot(ax=ax2)
    ax2.set_xlabel('Stopa lažno neuspešnih')
    ax2.set_ylabel('Stopa stvarno neuspešnih')
    ax2.set_title('ROC kriva neuspešnih')
    ax2.legend(loc='lower right')

    plt.show()

    return model


def ucitaj(data):
    X = data.iloc[:, 1:-1]
    y = data.loc[:, ['VTVF']]
    return X, y

pd.set_option('display.max_columns', None)
data1 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_prvi.csv", sep=',')
data2 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_prvi_disp.csv", sep=',')
data3 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi.csv", sep=',')
data4 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi_disp.csv", sep=',')
data5 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno.csv", sep=',')
data6 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno_disp.csv", sep=',')

X, y = ucitaj(data1)
vtvf_pojava = data1.iloc[:, -1]
X = X.join(y)
vtvf_pojava = vtvf_pojava.map(lambda x: bool(x))
X.iloc[:, -1] = vtvf_pojava

y = X.iloc[:, -1]

X[np.isnan(X)] = -100

pd.set_option('display.max_columns', None)


trening, test = train_test_split(X, random_state=0, test_size=0.3, stratify=y)

model = BayesianModel([(y, att) for att in X])

model.fit(
    data=trening,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=100,
    complete_samples_only=False
)

klasifikuj(model, test, y)

