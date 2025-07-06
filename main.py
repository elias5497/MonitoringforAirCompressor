import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Config ===
MaxExpNo = 10
labels = ['Bearing', 'Flywheel', 'Healthy', 'LIV', 'LOV', 'NRV', 'Piston', 'Riderbelt']
Faults = {label: idx for idx, label in enumerate(labels)}

# === Feature Labels ===
import PrimaryStatFeatures
import FFT_Module

data_columns_PrimaryStatFeatures = ['Mean', 'Min', 'Max', 'StdDv', 'RMS', 'Skewness', 'Kurtosis', 'CrestFactor', 'ShapeFactor']
data_columns_Target = ['Fault']

# === Data Loading ===
data = None
counter = -1

for label in labels:
    for ExpNo in range(1, MaxExpNo + 1):
        counter += 1
        file = f'D:/Summer term/EE656/Paper2/Data/{label}/preprocess_Reading{ExpNo}.txt'
        print(f"Loading file: {file}")
        X = np.loadtxt(file, delimiter=',')

        if (counter % 10 == 0):
            print(f'Loading files: {counter / (len(labels) * MaxExpNo) * 100:.2f}% completed')

        # Extract features
        StatFeatures = PrimaryStatFeatures.PrimaryFeatureExtractor(X)
        FFT_Features, data_columns_FFT_Features = FFT_Module.FFT_BasedFeatures(X)
        data_columns = data_columns_PrimaryStatFeatures + data_columns_FFT_Features + data_columns_Target

        # Initialize DataFrame after knowing all columns
        if data is None:
            data = pd.DataFrame(columns=data_columns)

        StatFeatures[0].extend(FFT_Features)
        StatFeatures[0].extend([Faults[label]])
        df_temp = pd.DataFrame(StatFeatures, index=[0], columns=data_columns)
        data = pd.concat([data, df_temp], ignore_index=True)

# === Normalization ===
from sklearn import preprocessing

normalization_status = 'RobustScaler'  # Change this to try others

input_data_columns = data_columns_PrimaryStatFeatures + data_columns_FFT_Features
input_data = data.drop(columns=['Fault'])

if normalization_status == 'Normalization':
    input_data = pd.DataFrame(preprocessing.normalize(input_data, norm='l2', axis=0), columns=input_data_columns)
elif normalization_status == 'StandardScaler':
    input_data = pd.DataFrame(preprocessing.StandardScaler().fit_transform(input_data), columns=input_data_columns)
elif normalization_status == 'MinMaxScaler':
    input_data = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(input_data), columns=input_data_columns)
elif normalization_status == 'RobustScaler':
    input_data = pd.DataFrame(preprocessing.RobustScaler().fit_transform(input_data), columns=input_data_columns)
elif normalization_status == 'Normalizer':
    input_data = pd.DataFrame(preprocessing.Normalizer().fit_transform(input_data), columns=input_data_columns)
elif normalization_status == 'WithoutNormalization':
    print('No normalization applied.')

target_data = pd.DataFrame(data['Fault'], columns=['Fault'], dtype=int)

# === Dimensionality Reduction (optional) ===
from sklearn.model_selection import train_test_split

DimReductionStatus = False  # Set to True if you want PCA

if DimReductionStatus:
    from sklearn import decomposition
    import KNN_Classifier
    import SVC_Classifier

    knn_scores = []
    svc_scores = []
    component_range = range(1, 31)

    for nComponents in component_range:
        pca = decomposition.PCA(n_components=nComponents)
        input_data_reduced = pca.fit_transform(input_data)

        x_train, x_test, y_train, y_test = train_test_split(
            input_data_reduced, target_data, test_size=0.3, random_state=42, stratify=target_data
        )

        test_accuracy_knn = KNN_Classifier.KNNClassifier(x_train, x_test, y_train, y_test)
        test_accuracy_svc = SVC_Classifier.SVCClassifier(x_train, x_test, y_train, y_test)

        knn_scores.append(test_accuracy_knn)
        svc_scores.append(test_accuracy_svc)

    # Plot PCA Accuracy Curves
    plt.figure(figsize=(10, 5))
    plt.plot(component_range, knn_scores, label='KNN Accuracy', marker='o')
    plt.plot(component_range, svc_scores, label='SVC Accuracy', marker='s')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs PCA Components')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Use final reduced data for classifiers below
    input_data = input_data_reduced

# Final train-test split used for all classifiers
x_train, x_test, y_train, y_test = train_test_split(
    input_data, target_data, test_size=0.3, random_state=42, stratify=target_data
)

# === Decision Tree Classifier ===
import DT_Classifier
DT_Classifier.DTClassifier(x_train, x_test, y_train, y_test)

# === Classifier Imports ===
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# === Classifier Optimization Flags ===
SVMOptStatus = False
KNNOptStatus = False
MLPOptStatus = False
DTOptStatus = False

import CLFOptimizer

# === Write optimal classifier parameters to file ===
file1 = open('Optimized Parameters for Classifiers.txt', 'w')
print('\nOptimized parameters for different classifiers:\n\n', file=file1)

if SVMOptStatus:
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        print(f'Optimizing SVC with kernel={kernel}')
        params, acc = CLFOptimizer.SVCOPT(kernel, x_train, x_test, y_train, y_test)
        print(f'SVC-{kernel}, Gamma={params[0]}, C={params[1]}, Accuracy={acc}', file=file1)

if KNNOptStatus:
    params, acc = CLFOptimizer.KNNOPT(x_train, x_test, y_train, y_test)
    print(f'KNN, neighbors={params}, Accuracy={acc}', file=file1)

if MLPOptStatus:
    params, acc = CLFOptimizer.MLPOPT(x_train, x_test, y_train, y_test)
    print(f'MLP, hidden layers={params}, Accuracy={acc}', file=file1)

if DTOptStatus:
    params, acc = CLFOptimizer.DTOPT(x_train, x_test, y_train, y_test)
    print(f'DT, max_depth={params[0]}, min_split={params[1]}, min_leaf={params[2]}, Accuracy={acc}', file=file1)

file1.close()

# === Classifier Comparison ===
CLFnames = [
    "SVC-linear", "K-Nearest Neighbors", "Multi-Layer Perceptron",
    "Decision Tree", "Random Forest", "Gaussian Process", "AdaBoost",
    "Naive Bayes", "QDA"
]

classifiers = [
    SVC(),
    KNeighborsClassifier(),
    MLPClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=1000, learning_rate=1),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

# === Write classifier results to file ===
f = open('ClassificationResults.txt', 'w')
print('\nComparison of classifiers performance:\n\n', file=f)

import ClassificationModule
ClassificationModule.Classifiers(CLFnames, classifiers, x_train, x_test, y_train, y_test)
f.close()
