import numpy as np
import random
# import copy
# from History import History


class Environment():
    def __init__(self, beta, horizon, queuesModel, arrModel, snrModel, history, lookBack = False, flatten = True):
        self.numberOfQueues = len(queuesModel)
        self.numberOfProperies = 2
        self.queuesModel = queuesModel
        self.arrModel = arrModel
        self.snrModel = snrModel
        self.packetArrived = [0]*len(queuesModel)
        self.packetDroped = [0]*len(queuesModel)
        self.packetSent = [0]*len(queuesModel)
        self.emptySent = [0]*len(queuesModel)
        self.channelLost = [0]*len(queuesModel)
        self.served = [0]*len(queuesModel)
        self.simTime = 1
        self.totalDiscounted = 0
        self.history = history
        self.lookBack = lookBack
        self.flatten = flatten
        self.beta = beta
        self.horizon = horizon
        return

    def newEpochInit(self):#initialize a new epoch
        self.packetArrived = [0]*self.numberOfQueues
        self.packetDroped = [0]*self.numberOfQueues
        self.packetSent = [0]*self.numberOfQueues
        self.emptySent = [0]*self.numberOfQueues
        self.channelLost = [0]*self.numberOfQueues
        self.served = [0]*self.numberOfQueues
        self.simTime = 1
        self.totalDiscounted = 0
        self.currentLength = self._initLength()
        return

    def get_current_fi(self):#return a feature vector of current states (used for GLIE)
        length = (np.array(self._normalize_length(self.currentLength))).reshape(self.numberOfQueues, 1)
        lossRate = (np.array(self.packetDroped)/self.simTime).reshape(self.numberOfQueues, 1)
        t = np.array([self.simTime/self.horizon]).reshape(1,1)
        c = np.concatenate((length, lossRate, t), axis = 0)
        if self.flatten:
            return c.reshape(2*self.numberOfQueues+1, 1)
        else:
            return c

    def get_batch_random(self,BATCH_SIZE): #random sampling the history to get a batch of feature
        historyBatch = self.history.randomSample(BATCH_SIZE)
        fi = self._historyToFi(historyBatch)
        assert fi['this'].shape == (2*self.numberOfQueues+1,BATCH_SIZE)
        assert fi['next'].shape == (2*self.numberOfQueues+1,BATCH_SIZE)
        assert fi['a'].shape == (BATCH_SIZE,2)
        assert fi['r'].shape == (1,BATCH_SIZE)
        return fi

    def get_batch_mostRecent(self,BATCH_SIZE): #get a batch of feature from most recent transtions
        historyBatch = self.history.recentSample(BATCH_SIZE)
        fi = self._historyToFi(historyBatch)
        assert fi['this'].shape == (2*self.numberOfQueues+1,BATCH_SIZE)
        assert fi['next'].shape == (2*self.numberOfQueues+1,BATCH_SIZE)
        assert fi['a'].shape == (BATCH_SIZE,2)
        assert fi['r'].shape == (1,BATCH_SIZE)
        return fi

    def multiSim(self, allAction): #Simlulate one transition for all users
        totalReward = 0
        preTransition_length = self._normalize_length(self.currentLength)
        preTransition_lossRate = self._getLossRate()
        if not self.lookBack:
            for i in range(self.numberOfQueues):
                if i ==  allAction:
                    action = 1
                else:
                    action = 0
                r = self._singleSim(i, action)
                totalReward += r
        self.totalDiscounted += totalReward*(self.beta**self.simTime)
        if self.simTime == self.horizon:
            print('total discounted reward is {}'.format(self.totalDiscounted))
        self._updateHistory([preTransition_length, preTransition_lossRate], allAction, totalReward, [self._normalize_length(self.currentLength), self._getLossRate()], self.simTime/self.horizon)
        self.simTime += 1
        return totalReward

    def _singleSim(self, queueIndex, oneAction): #act a transition for a single queue
        isDropped = False
        maxLen = self.queuesModel[queueIndex]
        deliver = self.snrModel[queueIndex]
        currentLeng = self.currentLength[queueIndex]
        if self.packetArrived[queueIndex] > 0:
            loss = self.packetDroped[queueIndex]/self.packetArrived[queueIndex]
        else:
            loss = 0
        queue_start = currentLeng
        if oneAction == 1: # Try to send
            self.served[queueIndex] += 1
            if queue_start == 0: # nothing to send
                queue_afterSend = 0
                self.emptySent[queueIndex] += 1
            elif random.uniform(0, 1)<deliver: #delivered
                queue_afterSend = queue_start - 1
                self.packetSent[queueIndex] += 1
            else:# not delivered, channel loss
                queue_afterSend = queue_start
                self.channelLost[queueIndex] += 1
        else: # not trying to send
            queue_afterSend = queue_start
        if random.uniform(0, 1)<self.arrModel[queueIndex]: # a new packet arrives
            self.packetArrived[queueIndex] += 1
            if queue_afterSend == maxLen: # queue already full, packet dropped
                self.packetDroped[queueIndex] += 1
                isDropped = True
                queue_afterArrive = queue_afterSend
            else:
                queue_afterArrive = queue_afterSend+1
        else: # nothing arrives
            queue_afterArrive = queue_afterSend
        ins_r = self._instantR([isDropped], type = "simpleLoss") #calculate instand reward
        self.currentLength[queueIndex] = queue_afterArrive #update queue length
        return ins_r

    def _updateHistory(self, S0, A, R, S1, T):
        historyItem = {}
        historyItem['S0'] = S0
        historyItem['A'] = A
        historyItem['R'] = R
        historyItem['S1'] = S1
        historyItem['T'] = T
        return self.history.put(historyItem)

    def _instantR(self, paramList, type):# a collection of possible instant reward function
        if type == 'simpleLoss':
            isDropped = paramList[0]
            if isDropped:
                return -10
            else:
                return 0

    def _initLength(self):# randomly initialize queue lengths
        if self.lookBack:
            return
        else:
            state = []
            for i in range(self.numberOfQueues):
                state.append(random.randrange(self.queuesModel[i]+1))
            return state

    def _normalize_length(self, length): # normalize queue lengths to be in [0, 1]
        normalized = []
        for i in range(self.numberOfQueues):
            normalized.append(length[i]/self.queuesModel[i])
        return normalized

    def _getLossRate(self):
        return [self.packetDroped[i]/self.simTime for i in range(self.numberOfQueues)]

    def _historyToFi(self, historyBatch):#Translate a history batch to a feature batch
        length = len(historyBatch)
        s_this = []
        s_next = []
        a = []
        r = []
        for i, d in enumerate(historyBatch):
            length_this = d['S0'][0]
            lossRate_this = d['S0'][1]
            t_this = [d['T']]
            singleS_this = length_this + lossRate_this + t_this
            s_this.append(singleS_this)
            length_next = d['S1'][0]
            lossRate_next = d['S1'][1]
            t_next= [d['T']+1/self.horizon]
            singleS_next = length_next + lossRate_next + t_next
            s_next.append(singleS_next)
            a.append([i, d['A']])
            r.append(d['R'])
        fi = {}
        fi['this'] = np.array(s_this).T
        fi['next'] = np.array(s_next).T
        fi['a'] = np.array(a)
        fi['r'] = np.array(r).reshape(1, length)
        return fi
