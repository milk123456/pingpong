import pickle
import numpy as np
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#%% data preprocessing
# I put this program in the same folder as MLGame/games/pingpong/ml
# you can edit path to get log folder
path = os.path.join(os.path.dirname(__file__), "..", "log")

allFile = os.listdir(path) # load log file
data_set = []

for file in allFile:
    with open(os.path.join(path, file), "rb") as f:
        data_set.append(pickle.load(f)) # load data in data_set

# initialize
x = np.array([1, 2, 3, 4, 5])  # feature
y = np.array([0]) # label

for data in data_set:
    Ball_x = []
    Ball_y = []
    Ball_speed_x = []
    Ball_speed_y = []
    direction = []
    temp = []   # record the frame hitting the platform
    y_temp = [] # record label

    for i, sceneInfo in enumerate(data["scene_info"][3:-1]): #get the feature you need
        # Frames.append(sceneInfo["frame"])
        Ball_x.append(sceneInfo["ball"][189])
        Ball_y.append(sceneInfo["ball"][44])
        Ball_speed_x.append(sceneInfo["ball_speed"][7])
        Ball_speed_y.append(sceneInfo["ball_speed"][-7])
        # Platform_1P.append(sceneInfo["platform_1P"])
        # Platform_2P_x.append(sceneInfo["platform_2P"][0])
        
        # Blocker_y.append(sceneInfo["blocker"][1])


    # for i in range(len(Ball_y)):
    #     if Ball_y[i] > 70 and Ball_y[i] < 80:
    #         temp.append(i)
    #     elif Ball_y[i] == 80:
    #         temp.append(i)
    # counter = 0
    for i in range(len(Ball_y)):
        if Ball_y[i] > 410 and Ball_y[i] < 420:
            temp.append(i)
        elif Ball_y[i] == 420:
            temp.append(i)
    counter = 0

    for i in temp:
        while counter <= i :
            y_temp.append(Ball_x[i])
            counter+=1

    Ball_x = np.array(Ball_x[:len(y_temp)]).reshape((len(y_temp), 1))
    Ball_y = np.array(Ball_y[:len(y_temp)]).reshape((len(y_temp), 1))
    Ball_speed_x = np.array(Ball_speed_x[:len(y_temp)]).reshape((len(y_temp), 1))
    Ball_speed_y = np.array(Ball_speed_y[:len(y_temp)]).reshape((len(y_temp), 1))

    

    for i in range(len(Ball_speed_x)):
        if Ball_speed_y[i][0] > 0 :
            if Ball_speed_x[i][0] > 0:
                direction.append(0)
            else :
                direction.append(1)
        else :
            if Ball_speed_x[i][0] > 0:
                direction.append(2)
            else:
                direction.append(3)

    direction = np.array(direction).reshape((len(direction), 1))
    x = np.vstack((x, np.hstack((Ball_x, Ball_y, direction, Ball_speed_x, Ball_speed_y))))
    y = np.hstack((y, np.array(y_temp)))

x = x[1::] #remove [1, 2, 3, 4, 5, 6]
y = y[1::] #remove [0]

#%% Decisiontree training model, Ref :  https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

model = DecisionTreeRegressor(max_depth = 300) # you can set any max_depth
model.fit(x, y)

#%% MLP training model, Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# model = MLPClassifier(random_state=2, max_iter=10000)
# # MLP hidden layer sizes default = 100
# # you can set any random_state to determine random number generation for weights and bias initialization
# # max_iter means the maximum number of iterations

# model.fit(x_train, y_train)

# # for training accuracy
# #print("Training set score: %f" % model.score(x_train, y_train))
# #yp_nn = model.predict(x_test)
# #acc = accuracy_score(yp_nn, y_test)

#%% save the model
path = os.path.join(os.path.dirname(__file__), "save")
if not os.path.isdir(path):
    os.mkdir(path)

with open(os.path.join(path, "model1.pickle"), 'wb') as f:
    pickle.dump(model, f)
