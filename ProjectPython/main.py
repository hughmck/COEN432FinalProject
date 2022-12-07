import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, roc_curve, roc_auc_score, classification_report, plot_roc_curve


def data_info():
    # Setting column names
    colnames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital_status",
                "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                "hours_per_week", "native_country", "income"]

    # Reading in both datasets and concatenating them
    df1 = pd.read_csv("adult.data", header=None)
    df2 = pd.read_csv("adult.test", skiprows=1, header=None)
    df = pd.concat([df1, df2])
    df.columns = colnames

    # Handing missing values and converting non-numerical values
    df = df.replace('?', np.NaN)
    df = df.dropna(axis=0)

    # Change label to binary; 1 if income greater than 50K; 0 if income lessthan or equal to 50K
    df["income"] = np.where(df["income"].str.contains(">50K"), 1, 0)

    # Getting distributions of income
    greater = (df["income"] == 1).sum()
    less = (df["income"] == 0).sum()

    plt.pie([greater, less], labels=['>50K', '<=50K'])
    plt.title("Distribution of Income")
    plt.show()

    return df

def PreProcessing(df):

    # For binary classification, change categorical to binary values (ie. One Hot Encoding)
    df = pd.get_dummies(df)

    # Getting X and y from input dataset
    y = df['income'].values
    df.pop("income")
    X = df.values

    # Splitting data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Normalizing the data
    scale = preprocessing.StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    return X_train, X_test, y_train, y_test

def KNearestNeighbour(X_train, y_train, X_test, y_test):

    # Model fitting with Grid Search and K-Cross Validation
    k_range = list(range(1,16))
    param_grid = dict(n_neighbors=k_range) # Search space

    grid = GridSearchCV(KNN(), param_grid, cv=10, scoring='accuracy', return_train_score=True, verbose=4)

    grid_search = grid.fit(X_train, y_train)
    best_params = grid_search.best_params_
    train_accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning using KNN is : {:.2f}%".format(train_accuracy) )

    # Checking accuracy on Test Data with KNN
    knn = KNN(**grid_search.best_params_)
    knn.fit(X_test, y_test)

    y_test_hat = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_hat) * 100
    print("Accuracy for our testing dataset with tuning using KNN is : {:.2f}%".format(test_accuracy))

    # Confusion Matrix
    metrics.accuracy_score(y_test, y_test_hat)
    confusion_matrix(y_test, y_test_hat)
    plot_confusion_matrix(knn, X_test, y_test)

    plt.savefig("ConfusionMatrixKNN.png", dpi=300)
    plt.show()

    return best_params

def DecisionTree(X_train, y_train, X_test, y_test):

  # Model fitting with Grid Search and K-Cross Validation
    param_grid = {'max_leaf_nodes': list(range(2, 15)),
                  'min_samples_split': [3, 4],
                  'max_depth': [4, 5, 6]}

    grid = GridSearchCV(DTC(), param_grid, cv=10, scoring='balanced_accuracy', return_train_score=True, verbose=4)

    grid_search = grid.fit(X_train, y_train)
    best_params = grid_search.best_params_
    train_accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(train_accuracy) )


    # Checking accuracy on Test Data with DTC
    dtc = DTC(**grid_search.best_params_)
    dtc.fit(X_test, y_test)

    y_test_hat = dtc.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_hat) * 100

    print("Accuracy for our testing dataset with tuning using DTC is : {:.2f}%".format(test_accuracy))

    # Confusion Matrix
    metrics.accuracy_score(y_test, y_test_hat)
    confusion_matrix(y_test, y_test_hat)
    plot_confusion_matrix(dtc, X_test, y_test)

    plt.savefig("ConfusionMatrixDTC.png", dpi=300)
    plt.show()

    return best_params


def SupportVectorMachine(X_train, y_train, X_test, y_test):
    param_grid = {'C': [0.1, 1],
                  'gamma': [1, 0.1, 0.001],
                  'kernel': ['linear']}
    #tested with cv=3 for as it took too log to run with cv=10
    grid = GridSearchCV(SVC(), param_grid, cv=10, scoring='accuracy', n_jobs=-1, return_train_score=True, verbose=4)
    grid_search = grid.fit(X_train, y_train)

    best_params = grid_search.best_params_
    train_accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning using SVC is : {:.2f}%".format(train_accuracy) )

    # Checking accuracy on Test Data with DTC
    svc = SVC(**grid_search.best_params_, probability=True)
    svc.fit(X_test, y_test)

    y_test_hat = svc.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_hat) * 100

    print("Accuracy for our testing dataset with tuning using SVC is : {:.2f}%".format(test_accuracy))

    # Confusion Matrix
    metrics.accuracy_score(y_test, y_test_hat)
    confusion_matrix(y_test, y_test_hat)
    plot_confusion_matrix(svc, X_test, y_test)

    plt.savefig("ConfusionMatrixSVC.png", dpi=300)
    plt.show()

    return best_params

def ROC(X_train, y_train, X_test, y_test):
    models = [KNN(),
              DTC(),
              SVC(probability=True)]

    perf = {}
    # Get ROC curves for different models to compare
    for model in models:
        fit = model.fit(X_train, y_train)
        y_test_prob = fit.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        auroc = roc_auc_score(y_test, y_test_prob)
        perf[type(model).__name__] = {'fpr': fpr, 'tpr': tpr, 'auroc': auroc}
    plt.clf()

    i = 0
    for model_name, model_perf in perf.items():
        plt.plot(model_perf['fpr'], model_perf['tpr'], label=model_name)
        plt.text(0.4, i + 0.1, model_name + ': AUC = ' + str(round(model_perf['auroc'], 2)))
        i += 0.1

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC in predicting Income')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="upper left")
    plt.show()
    plt.savefig("roc_curve.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    df = data_info()
    X_train, X_test, y_train, y_test = PreProcessing(df)
    knnResult = KNearestNeighbour(X_train, y_train, X_test, y_test)
    print(knnResult)
    dtcResult = DecisionTree(X_train, y_train, X_test, y_test)
    print(dtcResult)
    svcResult = SupportVectorMachine(X_train, y_train, X_test, y_test)
    print(svcResult)
    ROC(X_train, y_train, X_test, y_test)



