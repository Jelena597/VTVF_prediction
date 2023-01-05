import pandas as pd
from random import sample
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import networkx as nx
from pgmpy.models import BayesianNetwork
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

def ucitaj(data):
    X = data.iloc[:, 1:-1]
    y = data.loc[:, ['VTVF']]
    return X, y

pd.set_option('display.max_columns', None)
data3 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi.csv", sep=',')
data4 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi_disp.csv", sep=',')
data5 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno.csv", sep=',')
data6 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno_disp.csv", sep=',')
data7 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi_ICD.csv", sep=',')
data8 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_drugi_disp_ICD.csv", sep=',')
data9 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno_ICD.csv", sep=',')
data10 = pd.read_csv("C:\\Users\\jjelena\\Desktop\\ICD_podaci\\PodaciZaKlasifikaciju_ocisceno_disp_ICD.csv", sep=',')

X, y = ucitaj(data10)

pd.set_option('display.max_columns', None)
print(X.head())
print(X.shape)
print('********')

#quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
#X = quantile_transformer.fit_transform(X)

digitized_df = None
initialized = False

'''
X1 = X.iloc[:, 15:20]
X2 = X.iloc[:, 21:]

X1 = X.iloc[:, 10:16]
X2 = X.iloc[:, 17:]

X1 = X.iloc[:, 12:17]
X2 = X.iloc[:, 18:]

X1 = X.iloc[:, 9:14]
X2 = X.iloc[:, 15:]

X1 = X.iloc[:, 12:16]
X2 = X.iloc[:, 17:]


X1 = X.iloc[:, 9:13]
X2 = X.iloc[:, 14:]

X1 = X.iloc[:, 12:]

X1 = X.iloc[:, 10:]

X1 = X.iloc[:, 12:]

X1 = X.iloc[:, 9:]
'''
X1 = X.iloc[:, 9:]


# Kontinualne vrednosti i vrednosti sa velikim opsegom nalaze u kolonama 3, 4, 7, 9
preostali = []
#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16]
#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17]
#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14]
#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16]
#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13]

#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
col_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8]
for col_index in col_ind:
    try:
        col = X.columns[col_index]
        column_data = X[col]
        percentile_points = np.linspace(0,1,10) * 100
        bins = np.percentile(column_data.dropna(), percentile_points)
        digitized_data = np.digitize(column_data, bins).astype('int')
        col_df = pd.DataFrame(data=np.array([digitized_data]).T, columns=[col])
    #     col_df

        if not initialized:
            initialized = True
            digitized_df = col_df
        else:
            digitized_df = digitized_df.join(col_df)
    except:
        print(f'Failed to process column: {col}')
'''
print(X1.head())
print('********')
print(X2.head())
print('********')
print(digitized_df.head())
print("**********************")
'''

#dataAll = digitized_df.iloc[:, :15].join(X1).join(digitized_df.iloc[:, 16]).join(X2).join(y)
dataAll = digitized_df.join(X1).join(y)
print(dataAll.shape)
#print(dataAll.shape)

#est = PC(dataAll)
#best_model = est.estimate(scoring_method=BDeuScore(dataAll), tabu_length=500)
#best_model = est.estimate(scoring_method=BicScore(dataAll), tabu_length=500)
#best_model = est.estimate(scoring_method=K2Score(dataAll), tabu_length=500)
#best_model = est.estimate(scoring_method=BDsScore(dataAll), tabu_length=500)
#best_model = est.estimate(variant='stable', ci_test='pearsonr', return_type='dag')
#best_model = est.estimate(variant='stable', ci_test='chi_square', return_type='dag')

#est = HillClimbSearch(dataAll)
#best_model = est.estimate(scoring_method=BDeuScore(dataAll))
#best_model = est.estimate(scoring_method=BicScore(dataAll))
#best_model = est.estimate(scoring_method=K2Score(dataAll))
#best_model = est.estimate(scoring_method=BDsScore(dataAll))


est = TreeSearch(dataAll)
#best_model = est.estimate(estimator_type='chow-liu', edge_weights_fn='normalized_mutual_info')
best_model = est.estimate(estimator_type='chow-liu',  class_node='VTVF', edge_weights_fn='mutual_info')
#best_model = est.estimate(estimator_type='chow-liu',  class_node='VTVF', edge_weights_fn='adjusted_mutual_info')
#best_model = est.estimate(estimator_type='tan', class_node='VTVF', edge_weights_fn='normalized_mutual_info')
#best_model = est.estimate(estimator_type='tan',  class_node='VTVF', edge_weights_fn='mutual_info')
#best_model = est.estimate(estimator_type='tan',  class_node='VTVF', edge_weights_fn='adjusted_mutual_info')


G = nx.DiGraph(directed=True)
G.add_edges_from(best_model.edges)
pos = nx.kamada_kawai_layout(G)
nx.draw(G, with_labels=True, pos=pos)
print(best_model.nodes())
plt.show()

selected_nodes = list(set(np.array([[x,y] for (x,y) in best_model.edges]).ravel()))
#print("Selected nodes")
#print(selected_nodes)
reduced_dataset = dataAll.loc[:,selected_nodes]
#print("Reduced dataset")
#print(reduced_dataset)

#print("ZA UBACITI")
#print("ZA UBACITI")
#print("ZA UBACITI")
pom = reduced_dataset.loc[reduced_dataset.isin([10, 22, 35]).any(axis=1)]
#print(pom.index)
#print("ZA UBACITI")
#print("ZA UBACITI")
#print("ZA UBACITI")

n_ceo = reduced_dataset.shape[0]
train_size = 0.7
n_umanji = reduced_dataset.loc[reduced_dataset.isin([43, 76, 38, 55, 50, 49, 86, 10, 22, 35, 14, 47, 40, 27, 53, 34, 26, 71, 20, 42, 78, 32, 34]).any(axis=1)].shape[0]
X_train1 = reduced_dataset.loc[reduced_dataset.isin([43, 76, 55, 38, 50, 49, 86, 10, 22, 35, 14, 47, 40, 27, 53, 34, 26, 71, 20, 42, 78, 32, 34]).any(axis=1)]
reduced_dataset = reduced_dataset.drop(labels=pom.index)


n = n_ceo - n_umanji
#print("N")
#print(X_train1.shape)
#print(reduced_dataset.shape)
available_ind = [x for x in range(n)]
train_set_ind = sample(available_ind, int(train_size * n))
test_set_ind = list(set(available_ind) - set(train_set_ind))


X_train2 = reduced_dataset.iloc[train_set_ind,:]
#X_train = X_train1.join(X_train2)
X_train = pd.concat([X_train1, X_train2])
#print("X_train")
#print(X_train)
X_test = reduced_dataset.iloc[test_set_ind,:]
#print(X_test)

#print("*****************")
#print("*****************")
#print("*****************")
#print(X_train.shape)
#print(X_test.shape)
#print("*****************")
#print("*****************")
#print("*****************")




#Klasifikacija Bajesom
bn = BayesianNetwork()
bn.add_edges_from(best_model.edges)
bn.fit(data=X_train, estimator=MaximumLikelihoodEstimator)
print(bn.check_model())

correct = 0
total = 0
target_variable = 'VTVF'

inference = VariableElimination(bn)
y = X_test.loc[:, [target_variable]].values.ravel()
y_pred = []

for index, row in X_test.iterrows():
    evidence = {}
    for col in X_test.columns:
        if col != target_variable:
            evidence[col] = int(row[col])

    res = inference.map_query(variables=[target_variable], evidence=evidence, show_progress=False)
    y_pred.append(res[target_variable])

    total += 1
    if row[target_variable] == res[target_variable]:
        correct += 1

    print(f'{correct}/{total}: {correct / total}')




confM = confusion_matrix(y, y_pred)
print(confM)
sb.heatmap(confM,cmap='coolwarm',annot=True)
plt.show()
bn.fit(reduced_dataset)
print(bn.edges)
for e in bn.edges:
    if e[0]=="VTVF":
        print(e)
    if e[1]=="VTVF":
        print(e)