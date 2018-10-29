import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from lib.Environment import Environment
from lib.History import History

#Test Cases
NUM_EPOCH = 3000
HORIZON = 1000

USERCASE = 10
if USERCASE == 3:
    horizon = 1000
    queuesModel = [10,10,10]
    loss_thres = [0.02, 0.05, 0.05]
    arriveModel = [0.15, 0.3, 0.3]
    snrModel = [0.9, 0.9, 0.9]
elif USERCASE == 5:
    horizon = 1000
    queuesModel = [15,15,15,15,15,15,15,15,15,15]
    loss_thres = [0.02, 0.05, 0.05, 0.02, 0.05, 0.02, 0.05, 0.05, 0.02, 0.05,]
    arriveModel = [0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.15, 0.05, 0.05, 0.1]
    snrModel = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
elif USERCASE == 10:
    horizon = 1000
    queuesModel = [15,15,15,15,15,15,15,15,15,15]
    loss_thres = [0.02, 0.05, 0.05, 0.02, 0.05, 0.02, 0.05, 0.05, 0.02, 0.05,]
    arriveModel = [0.05, 0.05, 0.05, 0.05, 0.05, 0.35, 0.05, 0.05, 0.05, 0.05]
    snrModel = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
else:
    raise ValueError('invalid user number case!')

#Model hyperparameters
HIDDEN_LAYER = []

BATCH_SIZE = 64
POOLSIZE = 100000
RANDOM_BATCH = True
BETA = 0.9999

numberOfActions = USERCASE
numberOfQueues = USERCASE
featureSize = 2*numberOfQueues+1
numberOfNeurons = [featureSize] + HIDDEN_LAYER +[numberOfActions]

def qEstimate(R, X, parameters, BETA):
    idx = tf.constant([featureSize-1])
    simTime = tf.gather(X, idx)
    final = tf.to_float(tf.less(simTime, 1.0))
    return R + BETA* tf.multiply(tf.reduce_max(forward_prop(X, parameters), axis = 0), final)

def initialization_parameters(featureSize, numberOfNeurons):

    W1 = tf.get_variable("W1", [numberOfNeurons[1],numberOfNeurons[0]], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [numberOfNeurons[1],1], initializer = tf.zeros_initializer())
    # W2 = tf.get_variable("W2", [numberOfNeurons[1],numberOfNeurons[0]], initializer = tf.contrib.layers.xavier_initializer())
    # b2 = tf.get_variable("b2", [numberOfNeurons[1],1], initializer = tf.zeros_initializer())
    # W3 = tf.get_variable("W3", [numberOfNeurons[2],numberOfNeurons[1]], initializer = tf.contrib.layers.xavier_initializer())
    # b3 = tf.get_variable("b3", [numberOfNeurons[2],1], initializer = tf.zeros_initializer())
    # W4 = tf.get_variable("W4", [numberOfNeurons[3],numberOfNeurons[2]], initializer = tf.contrib.layers.xavier_initializer())
    # b4 = tf.get_variable("b4", [numberOfNeurons[3],1], initializer = tf.zeros_initializer())

    parameters = {'W1': W1,
                  'b1': b1,
                  # 'W2': W2,
                  # 'b2': b2,
                  # 'W3': W3,
                  # 'b3': b3,
                  # 'W4': W4,
                  # 'b4': b4
                  }
    return parameters

def forward_prop(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    # W2 = parameters['W2']
    # b2 = parameters['b2']
    # W3 = parameters['W3']
    # b3 = parameters['b3']
    # W4 = parameters['W4']
    # b4 = parameters['b4']

    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    # A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    # Z2 = tf.add(tf.matmul(W2, A1), b2)                                             # Z2 = np.dot(W2, a1) + b2
    # A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    # Z3 = tf.add(tf.matmul(W3, A2), b3)                                               # Z3 = np.dot(W3,Z2) + b3
    # A3 = tf.nn.relu(Z3)
    # Z4 = tf.add(tf.matmul(W4, A3), b4)
    End = Z1

    return End

def glie(fi, epsilon, parameters):# epsilon is explore prob
    if np.random.rand() < epsilon:
        return np.random.randint(numberOfActions)
    else:
        w = sess.run(z_glie, feed_dict= {x_glie: fi})
        # print(w.shape)
        # print(fi.shape)
        # input()
        # print(w)
        # print(np.argmax(w))
        a = np.argmax(w)
        # if a > numberOfActions:
        #     print(a)
        #     print(w.shape)
        #     input()
        return a

def sigmoid(x):
    return 1/(1+math.exp(-x))


if __name__ == "__main__":

    #tensors to model
    tf.reset_default_graph()
    parameters = initialization_parameters(featureSize, numberOfNeurons)
    x = tf.placeholder(tf.float32, [featureSize, None], name = "x")
    x_next = tf.placeholder(tf.float32, [featureSize, None], name = "x_next")
    actions_index = tf.placeholder(tf.int32, [BATCH_SIZE, 2], name = "actions_index")
    z = forward_prop(x, parameters)
    qsa = tf.transpose(tf.gather_nd(tf.transpose(z),actions_index))
    instant_r = tf.placeholder(tf.float32, [1, None], name = 'instant_r')
    qE = qEstimate(instant_r, x_next,parameters, BETA)
    cost = tf.reduce_mean(tf.squared_difference(qsa, qE))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
    x_glie = tf.placeholder(tf.float32, [featureSize,None], name = 'x_glie')
    z_glie = forward_prop(x_glie, parameters)



    init = tf.global_variables_initializer()



    with tf.Session() as sess:
        sess.run(init)

        hs = History(POOLSIZE)
        # historyPool.append(history)
        ev = Environment(BETA,HORIZON, queuesModel, arriveModel, snrModel, hs)
        history = ev.newEpochInit()
        for i in range(BATCH_SIZE*2):
            currentFi = ev.get_current_fi()
            action = glie(currentFi, 1, parameters)
            ev.multiSim(action)
        print('ready to learn!')
        fd_loss = open('loss.csv','w')
        fd_reward = open('reward.csv','w')
        input()

    ### learning phase
        historyCounter = BATCH_SIZE*2-1
        startPoint = BATCH_SIZE*2
        for reset in range(NUM_EPOCH):
            print("**************number of reset is {}*******************".format(reset))
            if reset > 0:
                startPoint = 0
                ev.newEpochInit()
                print(ev.currentLength)
                # input()
            for i in range(startPoint,HORIZON):
                historyCounter += 1
                if i%500 == 0 and i!=0:
                    # print(i)
                    # print(ev.simTime)
                    print('cost is {}'.format(cost_run))
                    fd_loss.write(str(cost_run)+'\n')
                # historyParsed = parseHistory(history)
                thisFi = ev.get_current_fi()
                explore_rate = max([500000/(500000+historyCounter), 0.1])
                action = glie(thisFi, explore_rate, parameters)
                ev.multiSim(action)
                if RANDOM_BATCH:
                    fi_dic = ev.get_batch_random(BATCH_SIZE)
                else:
                    fi_dic = ev.get_batch_mostRecent(BATCH_SIZE)
                # print(sess.run(z, feed_dict = {x: fi_dic['this']}))
                # print(fi_dic['a'])
                # print(sess.run(qsa, feed_dict = {x: fi_dic['this'], actions_index: fi_dic['a']}))
                # input()
                _, cost_run = sess.run([optimizer,cost], feed_dict = {x_next: fi_dic['next'], instant_r: fi_dic['r'], x: fi_dic['this'], actions_index: fi_dic['a']})
            print('packetDroped: {}'.format(ev.packetDroped))
            print('packetArrived: {}'.format(ev.packetArrived))
            print('packetSent: {}'.format(ev.packetSent))
            print('emptySent: {}'.format(ev.emptySent))
            print('channel lost: {}'.format(ev.channelLost))
            print('user served: {}'.format(ev.served))
            fd_reward.write(str(ev.totalDiscounted)+'\n')
        # fd.close()
