# Load libraries
from sklearn.ensemble import AdaBoostClassifier
# Import train_test_split function
from sklearn.tree import DecisionTreeClassifier
#Import scikit-learn metrics module for accuracy calculation
import numpy as np, os , time
from sklearn.model_selection import KFold
from validate import validation
from sklearn import tree as sktree

def enumarate_file(directory):
    return [filename for filename in sorted(os.listdir(directory))]

def prediction(x,y):
    hasil = []
    for index,xx in enumerate(x):
        # print(str(x[index])+" : "+str(y[index]))
        if(x[index] == y[index]):
            hasil.append(1)
        else:
            hasil.append(0)
    return hasil


if __name__ == '__main__':
    #Init
    # kelas = 3
    # data_awal = "data_iris_3.txt"

    kelas = 5
    # data_awal = "../data_awal"
    data_awal = "../data_2017-2019_5-label"
    data_hasil = "C4.5_data_hasil_5-label"

    # kelas = 10
    # # data_awal = "../data_awal"
    # data_awal = "../data_2017-2019_10-label"
    # data_hasil = "C4.5_data_hasil-10-label"

    max_depths = [20]

    for max_depth in max_depths:
        for file in enumarate_file(data_awal):
            subfolder = str(max_depth) + "-depth_"
            if not os.path.exists(data_hasil + "/" + subfolder):
                os.makedirs(data_hasil + "/" + subfolder)

            # new_file = open(data_hasil + "/" + subfolder + "/" + subfolder + "_" + file, 'w')

            # Load data Manual
            data = np.genfromtxt(data_awal + "/" + file, delimiter=",")
            hasil_predict = []

            X = []
            for i, x in enumerate(data):
                X.append(x[0:-kelas])

            Y = []
            for l in range(1, kelas + 1):
                y = []
                for i, x in enumerate(data):
                    y.append(int(x[-l]))
                Y.append(y)

            # FOLD
            # kf = KFold(n_splits=10,random_state=None, shuffle=True)
            kf = KFold(n_splits=10, shuffle=True)
            for train_index, test_index in kf.split(X):
                start = time.process_time()
                pred = []
                test = []
                for l in range(0, kelas):
                    # Split dataset into training set and test set
                    X_train, X_test = X[train_index[0]:train_index[-1]], X[test_index[0]:test_index[-1]]
                    y_train, y_test = Y[l][train_index[0]:train_index[-1]], Y[l][test_index[0]:test_index[-1]]

                    clf = sktree.DecisionTreeClassifier(criterion='entropy',max_depth=max_depth)
                    clf = clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    pred.append(y_pred)
                    test.append(y_test)

                predT = np.array(pred).T.tolist()
                testT = np.array(test).T.tolist()
                # print(predT) # Prediction
                # print(testT) # Test
                validation_result = validation(predT, testT)

                end = time.process_time()
                learning_time = str(end - start)
                print("Waktu ")
                print(start)
                print(end)
                print(validation_result + "|" + learning_time)
                # new_file.write(validation_result + "|" + learning_time + "\n")

            # new_file.close()

