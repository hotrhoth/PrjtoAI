import pandas as pd
import numpy  as np
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")


def loadAndShuffleData():
    # Load file .csv
    data = pd.read_csv("data/real_estate.csv")  
    # Random (1 -> 144) data
    data = data.sample(frac = 1).reset_index(drop = True)

    return data


def compressData(data):
    arrmin    = [0 for i in range(8)]
    arrmax    = [0 for i in range(8)]

    head = list(data.columns)

    for i in range(1, 8, 1):
        arrmin[i] = min(data[head[i]])
        arrmax[i] = max(data[head[i]])

    for i in range(414):
        for j in range(1,8,1):
            data.iat[i, j] = (data.iat[i, j] - arrmin[j]) / (arrmax[j] - arrmin[j])


def linearRegressionModel(train_data_x, train_data_y, test_data_x, test_data_y):
    train_linear = linear_model.LinearRegression()
    train_linear.fit(train_data_x, train_data_y)
    score_trained = train_linear.score(test_data_x, test_data_y)

    return score_trained

def mean_squared_error(y_true, y_predicted):

    cost = np.sum((y_true-y_predicted)**2) / len(y_true)

    return cost

def gradient_descent(x, y, iterations = 1000000000, learning_rate = 0.5, stopping_threshold = 1e-12):
    current_weight = 0.1
    current_bias   = 0.01
    iterations     = iterations
    learning_rate  = learning_rate
    n              = float(len(x))
     
    costs         = []
    weights       = []
    previous_cost = None
     
    for i in range(iterations):
        y_predicted = (current_weight * x) + current_bias
         
        current_cost = mean_squared_error(y, y_predicted)
 
        if previous_cost and abs(previous_cost-current_cost) <= stopping_threshold:
            break
         
        previous_cost = current_cost
 
        costs.append(current_cost)
        weights.append(current_weight)
         
        # Calculating the gradients
        weight_derivative =  - (2 / n) * sum(x * (y - y_predicted))
        bias_derivative   =  - (2 / n) * sum(y - y_predicted)
         
        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias   = current_bias   - (learning_rate * bias_derivative)

    return current_weight, current_bias


def main():
    data = loadAndShuffleData()

    compressData(data)

    # 90% train | 10 % test (total = 414)
    x = 414 * 90 // 100 
    train_data_x = data.iloc[ : x , 1: 7]
    train_data_y = data.iloc[ : x , 7:  ]
    test_data_x  = data.iloc[x:   , 1: 7]
    test_data_y  = data.iloc[x:   , 7:  ]

    train_data_x.to_csv("data/train/data.csv" , index = False)
    train_data_y.to_csv("data/train/label.csv", index = False)
    test_data_x .to_csv("data/test/data.csv"  , index = False)
    test_data_y .to_csv("data/test/label.csv" , index = False)

    w = np.array([0, 0, 0, 0, 0, 0], dtype = float)
    b = np.array([0, 0, 0, 0, 0, 0], dtype = float)    

    for i in range(6):
        w[i], b[i] = gradient_descent(train_data_x.iloc[:,i : i + 1].to_numpy(), train_data_y.to_numpy())

    w_tmp = [x/6 for x in w]
    print("w[] = ", w_tmp)
    b_tmp = sum(b) / 6
    print("b = ", b_tmp)

    y_predicted = []
    
    for i in range(42):
        y = 0
        for j in range(6):
            y += test_data_x.iat[i, j] * w_tmp[j]
        y += b_tmp
        y_predicted.append(y)

    y_predicted = pd.DataFrame(y_predicted)

    cost = 0

    for i in range(42):
        cost += (test_data_y.iat[i, 0]-y_predicted.iat[i, 0])**2

    print(cost/42)

    ### chon cac tp gradient, de nho nhat co the

    # linear_score = linearRegressionModel(train_data_x, train_data_y, test_data_x, test_data_y)
    # print('Linear Regression score = ', linear_score)

if __name__ == "__main__":
    main()