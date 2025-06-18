import numpy as np


def getBreaths(df):
    minThreshold = 0.001
    mult = 1e-2
    
    signal = list(df.breathingSignal)
    
    time_diff = df['timestamp'].diff()
    time_diff.map(lambda x: x.total_seconds()).mean()
    
    window_size = int((30 / time_diff.dropna().apply(lambda x: x.total_seconds()).mean()) // 2)
    threshs = calculateThresholdLevels(list(signal), window_size, window_size, mult, False)
    posThresh = threshs[:, 0]
    negThresh = threshs[:, 1]

    times = calculateBreathTimes(list(signal), posThresh, negThresh, minThreshold, False)

    total = set()
    minBreathLength = float("inf")
    maxBreathLength = float("-inf")
    for i in range(0, len(times)):
        vals = times[i]
        for j in range(0, len(vals)-1):
            start, end = vals[j], vals[j+1]
            minBreathLength = min(minBreathLength, end-start+1)
            maxBreathLength = max(maxBreathLength, end-start+1)
            for k in range(start, end+1):
                total.add(k)

    f = list(df.breathingSignal.dropna())
    a = f"Uses Breath From {len(total)}/{len(f)} = {round((len(total)/len(f)) * 100, 2)}% Signal"
    b = f"Max Breath Length: {maxBreathLength} points. Min Breath Length: {minBreathLength} points"
    print(a)
    print(b)
        
    return times

def countLocalMaximas(values):
    count = 0
    if len(values) < 3:
        return 1
    if len(values) > 1 and values[0] > values[1]:
        count += 1
    if len(values) > 1 and values[-1] > values[-2]:
        count += 1
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            count += 1
    return count

def countLocalMinimas(values):
    count = 0
    if len(values) < 3:
        return 1
    if len(values) > 1 and values[0] < values[1]:
        count += 1
    if len(values) > 1 and values[-1] < values[-2]:
        count += 1
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            count += 1
    return count

def calculateBreathTimes(signal, posThresholds, negThresholds, minThreshold, zeroCrossingBreathStart):
    
    def breathTimes(startIndex, endIndex):

        def setInitialState(startValue, posThreshold, negThreshold):
            if startValue < negThreshold:
                state = LOW
            elif startValue > posThreshold:
                state = HIGH
            else:
                state = MID_UNKNOWN
            return state
    
        state = setInitialState(signal[startIndex], posThresholds[startIndex], negThresholds[startIndex])
        times = []
    
        for i in range(startIndex + 1, endIndex + 1):
            posThreshold = posThresholds[i]
            negThreshold = negThresholds[i]
            if state == LOW and signal[i] > negThreshold:
                state = MID_RISING
            elif state == HIGH and signal[i] < posThreshold:
                state = MID_FALLING
            elif (state == MID_RISING or state == MID_UNKNOWN) and signal[i] > posThreshold:
                state = HIGH
            elif (state == MID_FALLING or state == MID_UNKNOWN) and signal[i] < negThreshold:
                state = LOW
                times.append(i)

        if zeroCrossingBreathStart:
            zeroCrossingBreathTimes = []
            for t in times:
                for i in range(t,-1,-1):
                    if signal[i] >= 0:
                        zeroCrossingBreathTimes.append(i)
                        break
            return zeroCrossingBreathTimes
        else:
            return times

    LOW, MID_FALLING, MID_UNKNOWN, MID_RISING, HIGH = range(5)

    
    invalidated = np.ones(np.shape(signal), dtype=bool)
    for i in range(len(invalidated)):
        if posThresholds[i] > minThreshold or negThresholds[i] < -minThreshold:
            invalidated[i] = False
    

    minIslandLength = 0
    islandLimits = findIslandLimits(invalidated, minIslandLength)
    
    times = []
    for (start, end) in islandLimits:
        bt = breathTimes(start, end)
        if len(bt) > 0:
            times.append(bt)

    return times

def calculateThresholdLevels(signal, rmsBackwardLength, rmsForwardLength, rmsMultiplier, symmetrical):
    result = nans((len(signal), 2))
    
    if not symmetrical:
        
        #fill sum of squares buffers
        posValues = []
        negValues = []
        windowLength = rmsBackwardLength + rmsForwardLength
        if len(signal) < windowLength:
            return result
        
        lastBananaIndex = np.nan
            
        for i in range(windowLength - 1):
            if signal[i] >= 0:
                posValues.append(signal[i])
            elif signal[i] < 0:
                negValues.append(signal[i])
            else: # if nan
                lastBananaIndex = i
                
        posArray = np.array(posValues)
        negArray = np.array(negValues)
        
        sumOfSquaresPos = np.sum(posArray**2)
        posCount = len(posArray)
        sumOfSquaresNeg = np.sum(negArray**2)
        negCount = len(negArray)
        
        for i in range(0, len(signal)):
            if i < rmsBackwardLength or i >= len(signal) - rmsForwardLength:
                posResult = np.nan
                negResult = np.nan
            else:
                newValue = signal[i+rmsForwardLength-1]
                if np.isnan(newValue):
                    lastBananaIndex = i+rmsForwardLength-1
                else:
                    if newValue >= 0:
                        sumOfSquaresPos += newValue**2
                        posCount += 1
                    elif newValue < 0:
                        sumOfSquaresNeg += newValue**2
                        negCount += 1
                
                if not np.isnan(lastBananaIndex) and i - lastBananaIndex <= rmsBackwardLength:
                    posResult = np.nan
                    negResult = np.nan
                else:
                    posResult = np.sqrt(sumOfSquaresPos / posCount) * rmsMultiplier
                    negResult = -np.sqrt(sumOfSquaresNeg / negCount) * rmsMultiplier
                
                oldValue = signal[i-rmsBackwardLength]
                
                if oldValue >= 0:
                    sumOfSquaresPos -= oldValue**2
                    posCount -= 1
                elif oldValue < 0:
                    sumOfSquaresNeg -= oldValue**2
                    negCount -=1
            result[i,0] = posResult
            result[i,1] = negResult
            
        return result
    
    else:
        #fill sum of squares buffers
        allValues = []
        windowLength = rmsBackwardLength + rmsForwardLength
        if len(signal) < windowLength:
            return result
        
        #print "signal length: " + str(len(signal))
        #print "windowLength: " + str(windowLength)
        #print "backward length: " + str(rmsBackwardLength)
        #print "forward length: " + str(rmsForwardLength)
        
        lastBananaIndex = np.nan
        
        for i in range(windowLength - 1):
            if not np.isnan(signal[i]):
                allValues.append(signal[i])
            else:
                lastBananaIndex = i
        allArray = np.array(allValues)
        
        sumOfSquaresAll = np.sum(allArray**2)
        allCount = len(allArray)
        
        for i in range(0, len(signal)):
            if i < rmsBackwardLength or i >= len(signal) - rmsForwardLength:
                allResult = np.nan
            else:
                newValue = signal[i+rmsForwardLength-1]
                if np.isnan(newValue):
                    lastBananaIndex = i+rmsForwardLength-1
                else:
                    sumOfSquaresAll += newValue**2
                    allCount += 1
                
                if not np.isnan(lastBananaIndex) and i - lastBananaIndex <= rmsBackwardLength:
                    allResult = np.nan
                else:
                    allResult = np.sqrt(sumOfSquaresAll / allCount) * rmsMultiplier
                
                oldValue = signal[i-rmsBackwardLength]
                if not np.isnan(oldValue):
                    sumOfSquaresAll -= oldValue**2
                    allCount -= 1
                    
            result[i,0] = allResult
            result[i,1] = -allResult
        #figure()
        #plot(signal)
        #plot(result)
        #show()
        return result
    

def nans(dims):
    a = np.empty(dims)
    a[:] = np.nan
    return a

''' Find the RMS value of an input signal in array form. '''
def rms(signal):
    return np.sqrt(np.mean(signal**2))

def rmsHamming(signal):
    squares = signal**2
    weights = np.hamming(len(signal))
    weightedSum = 0.0
    weightsSum = 0.0

    for i in range(len(signal)):
        weightedSum += squares[i] * weights[i]
        weightsSum += weights[i]

    return np.sqrt(weightedSum / weightsSum)

''' Find islands of defined values in a signal that may contain NaNs. '''
def findIslandLimits(signal, minIslandLength=0, minIslandGap=0):

    islands = []

    start = None
    end = None
    foundIsland = False

    for i in range(len(signal)):
        if not signal[i]:
            if start == None:
                start = i
            else:
                end = i + 1
                if i == len(signal) - 1:
                    foundIsland = True
        else:
            if start != None:
                if end != None:
                    foundIsland = True
                else:
                    start = None

        if foundIsland:
            if (minIslandGap > 0) and (len(islands) > 0):
                prevIslandStart = islands[-1][0]
                prevIslandEnd = islands[-1][1]
                islandGap = start - prevIslandEnd - 1
                if islandGap < minIslandGap:
                    # merge the new island with the previous one
                    islands[-1] = ((prevIslandStart, end))
                else:
                    islands.append((start, end))
            else:    
                islands.append((start, end))

            start = None
            end = None
            foundIsland = False
            
    # now return only the islands that are long enough
    longIslands = []
    for island in islands:
        if (island[1] - island[0]) >= minIslandLength:
            longIslands.append(island)

    return longIslands
    

