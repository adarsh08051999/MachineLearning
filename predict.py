# To Run python predict.py train_X_knn.csv
# It's recommended to use 70-80% of the available data as train set and the rest for validation.
import numpy as np
import csv
import sys
import statistics
import math

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""
def compute_ln_norm_distance(vector1, vector2, n):
    sum1=0
    i=0
    while i<len(vector1):
       sum1=sum1+pow(abs(vector1[i]-vector2[i]),n)
       i=i+1
    return pow(sum1,1/n)


def find_k_nearest_neighbors(train_X, test_example, k, n):
    L=list()
    for i in range(0,len(train_X)):
        L.append([compute_ln_norm_distance(test_example,train_X[i], n),i])
    L.sort()
    M=list()
    for i in range(0,k):
        M.append(L[i][1])
    return M


def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y = []
    for test_elem_x in test_X:
        top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k, n)
        top_knn_labels = []

        for i in top_k_nn_indices:
            top_knn_labels.append(train_Y[i])
        Y_values = list(set(top_knn_labels))

        max_count = 0
        most_frequent_label = -1
        for y in Y_values:
            count = top_knn_labels.count(y)
            if (count > max_count):
                max_count = count
                most_frequent_label = y

        test_Y.append(most_frequent_label)
    return test_Y


def calculate_accuracy(predicted_Y, actual_Y):
    from sklearn.metrics import f1_score
    return f1_score(actual_Y[:len(predicted_Y)], predicted_Y, average = 'weighted')

    count=0
    for i in range(0,len(predicted_Y)):
        if predicted_Y[i]==actual_Y[i]:
            count=count+1
    return count/len(predicted_Y)


def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n):
    m=math.floor(len(train_X)*(100-validation_split_percent)/100)
    ans= -1
    accuracy= -1
    for k in range(1,m+1):
        predicted_Y=classify_points_using_knn(train_X[0:m], train_Y, train_X[m:], k, n)
        temp=calculate_accuracy(predicted_Y, train_Y)
        if(temp>accuracy):
            accuracy=temp
            ans=k
    return ans


def import_data(test_X_file_path,x):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=x)
    return test_X


def predict_target_values(test_X):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    test1_X=import_data("./train_X_knn.csv",1)
    train_Y=import_data("./train_Y_knn.csv",0)
    n=4
    #print(len(train_Y),test_X.shape)
    k = get_best_k_using_validation_set(test1_X,train_Y,75,n)
    #print(k)
    predicted_y = classify_points_using_knn(test1_X,train_Y,test_X,k,n)
    #print(calculate_accuracy(predicted_y, train_Y))
    #f i am getting F1 score of 1.0 at n=4 lucky as f
    predicted_y=np.array(predicted_y,dtype="int16")
    return predicted_y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path,1)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv")