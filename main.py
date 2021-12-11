
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from MFCC import make_MFCC
import os



def convert_data(MFCC_all):
    # In this function we convert MFCC for all 6 input voices to
    # desired format which the question asked for , X is input points
    # which is 30 * num_ceps and y is label of each point
    X = np.empty([0, np.shape(MFCC_all[0])[1] * 30])
    Y = np.empty([0])                                # array for storing label of each point which is [1 to 6]
    for label, row in enumerate(MFCC_all):
        pnt = []
        for i in range(10):
            index = np.random.choice(row.shape[0], size=30, replace=False)
            pnt.append(row[index].reshape(-1))
        X = np.concatenate([X, pnt])
        Y = np.concatenate([Y, (label + 1) * np.ones([(i + 1)])])
    return [X, Y]



def draw_dim_analysis(x, y, title, name):
    # this function is used to draw 2D plots for result's of LDA or PCA
    # x : input points
    # y : label of each point
    # title : title of each plot
    # name : name for saving output image
    colors = ["navy", "turquoise", "darkorange", 'red', 'green', 'blue']
    target_names = np.array(["phoneme 1", "phoneme 2", 'phoneme 3', 'phoneme 4', 'phoneme 5', 'phoneme 6'])
    fig = plt.figure(figsize=(9, 9))
    for i, target in enumerate(target_names):
        plt.scatter(x[y == (i + 1), 0], x[y == (i + 1), 1],color=colors[i], label=target)
    plt.title(title, fontsize=16)
    plt.xlabel('Component 1', fontsize=15)
    plt.ylabel('Component 2', fontsize=15)
    plt.legend(loc="best", shadow=False)
    plt.grid(linestyle=':', linewidth=0.9)
    fig.savefig(name + '.png', dpi=5 * fig.dpi)
    plt.close()


def draw_knn_analysis(classifier, x, y, title, name):
    # x : input points
    # y : label of each point
    # title : title of each plot

    target_names = np.array(["phoneme 1", "phoneme 2", 'phoneme 3', 'phoneme 4', 'phoneme 5', 'phoneme 6'])
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = x[:, 0].min() * 1.1, x[:, 0].max() * 1.1
    y_min, y_max = x[:, 1].min() * 1.1, x[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    # Z : mesh between different areas
    Z = (classifier.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape) 

    fig = plt.figure(figsize=(9, 9))
    plt.contour(xx, yy, Z, colors=6 * ['black'])
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=target_names[np.int32(y) - 1])
    plt.title(title, fontsize=16)
    plt.xlabel('Component 1', fontsize=15)
    plt.ylabel('Component 2', fontsize=15)
    plt.legend(loc="best", shadow=False)
    plt.grid(linestyle=':', linewidth=0.9)
    fig.savefig(name + '.png', dpi=5 * fig.dpi)
    plt.close()

# make directorys for saving results
try:os.mkdir('speaker1_result')
except: pass
try:os.mkdir('speaker2_result')
except: pass


#****************************#
#********* speaker 1 ********#
#****************************#


MFCC1_all = make_MFCC('S01', append_delta=True)
[X, Y] = convert_data(MFCC1_all)
X_PCA = PCA(n_components=2).fit_transform(X)
X_LDA = LinearDiscriminantAnalysis(n_components=2).fit(X, Y).transform(X)
clf_PCA = neighbors.KNeighborsClassifier(3)
clf_PCA.fit(X_PCA, Y)
clf_LDA = neighbors.KNeighborsClassifier(3)
clf_LDA.fit(X_LDA, Y)
draw_dim_analysis(x=X_PCA, y=Y, title="speaker1 : PCA", name='speaker1_result/S01_PCA')
draw_dim_analysis( x=X_LDA, y=Y, title="speaker1 : LDA for MFCC's with delta", name='speaker1_result/S01_LDA_delta')
draw_knn_analysis( clf_PCA, X_PCA, Y, title="speaker1 : KNN for PCA", name='speaker1_result/S01_knn_PCA')
draw_knn_analysis( clf_LDA, X_LDA, Y, title="speaker1 : KNN for LDA : MFCC's with delta", name='speaker1_result/S01_knn_LDA_delta')

#****** without delta in MFCC *****#
MFCC1_all = make_MFCC('S01', append_delta=False)
[X, Y] = convert_data(MFCC1_all)
X_LDA = LinearDiscriminantAnalysis(n_components=2).fit(X, Y).transform(X)
clf_LDA = neighbors.KNeighborsClassifier(3)
clf_LDA.fit(X_LDA, Y)
draw_dim_analysis( x=X_LDA, y=Y, title="speaker1 : LDA for MFCC's without delta", name='speaker1_result/S01_LDA_no_delta')
draw_knn_analysis( clf_LDA, X_LDA, Y, title="speaker1 : KNN for LDA : MFCC's without delta", name='speaker1_result/S01_knn_LDA_no_delta')


#****************************#
#********* speaker 2 ********#
#****************************#
MFCC2_all = make_MFCC('S02', append_delta=True)
[X, Y] = convert_data(MFCC2_all)
X_PCA = PCA(n_components=2).fit_transform(X)
X_LDA = LinearDiscriminantAnalysis(n_components=2).fit(X, Y).transform(X)
clf_PCA = neighbors.KNeighborsClassifier(3)
clf_PCA.fit(X_PCA, Y)
clf_LDA = neighbors.KNeighborsClassifier(3)
clf_LDA.fit(X_LDA, Y)
draw_dim_analysis( x=X_PCA, y=Y, title="speaker2 : PCA for MFCC", name='speaker2_result/S02_PCA')
draw_dim_analysis( x=X_LDA, y=Y, title="speaker2 : LDA for MFCC's with delta", name='speaker2_result/S02_LDA_delta')
draw_knn_analysis( clf_PCA, X_PCA, Y, title="speaker2 : KNN for PCA ", name='speaker2_result/S02_knn_PCA')
draw_knn_analysis( clf_LDA, X_LDA, Y, title="speaker2 : KNN for LDA : MFCC's with delta", name='speaker2_result/S02_knn_LDA_delta')

#****** without delta in MFCC *****#
MFCC1_all = make_MFCC('S02', append_delta=False)
[X, Y] = convert_data(MFCC1_all)
X_LDA = LinearDiscriminantAnalysis(n_components=2).fit(X, Y).transform(X)
clf_LDA = neighbors.KNeighborsClassifier(3)
clf_LDA.fit(X_LDA, Y)
draw_dim_analysis( x=X_LDA, y=Y, title="speaker2 : LDA for MFCC's without delta", name='speaker2_result/S02_LDA_no_delta')
draw_knn_analysis( clf_LDA, X_LDA, Y, title="speaker2 : KNN for LDA : MFCC's without delta", name='speaker2_result/S02_knn_LDA_no_delta')
