import random

class History:
    def __init__(self, maxLen):
        self.pool = []
        self.maxLen = maxLen
        self.replacePointer = 0
        self.currentLen = 0

    def put(self, historyItem): # put a transition history event into the pool
        if self.currentLen < self.maxLen:
            self.pool.append(historyItem)
            self.currentLen += 1
        else:
            self.pool[self.replacePointer] = historyItem
            self.replacePointer += 1
            if self.replacePointer >= self.maxLen:
                self.replacePointer = 0
        return

    def randomSample(self, sampleSize): # randomly sample history pool
        return random.sample(self.pool, sampleSize)

    def clearAll(self): # clear everything
        self.pool = []
        self.maxLen = maxLen
        self.replacePointer = 0
        self.currentLen = 0
        return

    def recentSample(self, sampleSize): # return most recents from the pool
        if self.currentLen< self.maxLen:
            if self.currentLen < sampleSize:
                print('History error: not enough samples')
                return 
            else:
                return self.pool[-1*sampleSize:]
        else:
            r = []
            for i in range(self.replacePointer - sampleSize, self.replacePointer):
                r.append(self.pool[i])
            return r
