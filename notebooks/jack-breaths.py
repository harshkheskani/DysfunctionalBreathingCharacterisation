# Jack's Util file

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


# Code from Jack Taylor

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

def generate_RRV(sliced):
    sliced = sliced.dropna()
    if sliced.size == 0:
        return np.nan
    breathingSignal = sliced.values
    N = breathingSignal.shape[-1]
    y = breathingSignal
    yf = np.fft.fft(y)
    yff = 2.0/N * np.abs(yf[:N//2])
    temp_DCnotremov = yff
    if len(temp_DCnotremov) == 0 or len(temp_DCnotremov) == 1: 
        return 0.0
    else:
        DC = np.amax(temp_DCnotremov)
        maxi = np.argmax(temp_DCnotremov)
        temp_DCremov = np.delete(temp_DCnotremov, maxi)
        H1 = np.amax(temp_DCremov)
        return 100-(H1/DC)*100

def getBreaths(df):
    minThreshold = 0.001
    mult = 0.0125
    
    signal = list(df.breathingSignal)
    
    time_diff = df['timestamp'].diff()
    time_diff.map(lambda x: x.total_seconds()).mean()
    
    window_size = int((20 / time_diff.dropna().apply(lambda x: x.total_seconds()).mean()) // 2)
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


def mode(l):
    if len(l) == 0:
        return np.NaN, {}, []
    
    sortedRoundedArray = np.sort(np.around(l))
    dict = {}
    dist = np.zeros(sortedRoundedArray[-1] + 1)
    maxCount = 0
    for e in sortedRoundedArray:
        dist[e] += 1
        if e in dict:
            newCount = dict[e] + 1
            dict[e] = newCount
        else:
            newCount = 1
            dict[e] = newCount
            
        if newCount > maxCount:
                maxCount = newCount
    
    if maxCount > 0:
        l = []
        for e in dict:
            if dict[e] == maxCount:
                l.append(e)
        sorted = np.sort(l)
        return sorted[len(sorted) // 2], dict, dist
                
    else:
        return np.NaN, dict, dist
    

def extractFeatures(df):
    times = getBreaths(df)

    areas = []
    extremas = []
    peakRespiratoryFlows = []
    types = []
    durations = []
    activityLevels = []
    activityTypes = []
    starts = []
    ends = []
    
    activityLevel = np.array(df.activityLevel)
    activityType = np.array(df.activityType)
    signal = np.array(df.breathingSignal)
    timestamps = list(df.timestamp)

    for i in range(0, len(times)):
        if i % 25 == 0:
            print(f"{i}/{len(times)}... ", end=" ")
        vals = times[i]
        
        for j in range(0, len(vals)-1):
            start, end = vals[j], vals[j+1]
            flag = False
            breath = signal[start:end+1]
            breakPoint = start
            for k, val in enumerate(breath):
                if val >= 0.005: # arbitrary but to remove noise...
                    breakPoint = start + k
                    break

            # compute inhalation
            inhalation, inhalation_times = signal[start:breakPoint], timestamps[start:breakPoint]
            exhalation, exhalation_times = signal[breakPoint:end+1], timestamps[breakPoint:end+1]
                    
            level = activityLevel[start:end+1].mean()
            modeType = mode(activityType[start:end+1])[0]
            
            # compute inhalation
            if len(inhalation) > 1:
                peak = max(abs(np.array(inhalation)))
                extrema = countLocalMaximas(inhalation)
                dx = (inhalation_times[-1]-inhalation_times[0]).total_seconds() / len(inhalation)
                area = abs(np.trapezoid(y=inhalation,dx=dx))
                duration = (inhalation_times[-1]-inhalation_times[0]).total_seconds()
                
                areas.append(area)
                extremas.append(extrema)
                peakRespiratoryFlows.append(peak)
                types.append("Inhalation")
                durations.append(duration)
                activityLevels.append(level)
                activityTypes.append(modeType)
                starts.append(inhalation_times[0])
                ends.append(inhalation_times[-1])

            if len(exhalation) > 1:
                peak = max(abs(np.array(exhalation)))
                extrema = countLocalMinimas(exhalation)    
                dx = (exhalation_times[-1]-exhalation_times[0]).total_seconds() / len(exhalation)
                area = abs(np.trapezoid(y=exhalation,dx=dx))  
                duration = (exhalation_times[-1]-exhalation_times[0]).total_seconds()
                
                areas.append(area)
                extremas.append(extrema)
                peakRespiratoryFlows.append(peak)
                types.append("Exhalation")
                durations.append(duration)
                activityLevels.append(level)
                activityTypes.append(modeType)
                starts.append(exhalation_times[0])
                ends.append(exhalation_times[-1])

    return pd.DataFrame(data={"type": types, "area": areas, "peakRespiratoryFlow": peakRespiratoryFlows, "extremas": extremas, "duration": durations, "meanActivityLevel": activityLevels, "modeActivityType": activityTypes, "startTimestamp": starts, "endTimestamp": ends})


def getRegularity(df):
    # get distance to 1st PC for area, PRF only --> makes rapid shallow in feature level
    # make it temporal by adding distance to PC from all 3 as a seperaate PCA raansform
    scaler = MinMaxScaler()
    columns = ['area', 'peakRespiratoryFlow']
    df_normalized = scaler.fit_transform(df[columns])
    pca = PCA(n_components=1)  
    pca.fit(df_normalized)
    df_pca = pca.transform(df_normalized)
    first_principal_component = pca.components_[0]
    te = np.linalg.norm(df_normalized - np.outer(df_normalized.dot(first_principal_component), first_principal_component), axis=1)

    scaler = MinMaxScaler()
    
    # Okay so we are looking at the resapmled breathing rate
    columns = ['area', 'peakRespiratoryFlow', 'BR_mean']
    df_normalized = scaler.fit_transform(df[columns])
    pca = PCA(n_components=3)  
    pca.fit(df_normalized)
    df_pca = pca.transform(df_normalized)
    
    first_principal_component = pca.components_[0]
    second_principal_component = pca.components_[1]
    third_principal_component = pca.components_[2]
    
    distances_to_first_component = np.linalg.norm(df_normalized - np.outer(df_normalized.dot(first_principal_component), first_principal_component), axis=1)
    distances_to_second_component = np.linalg.norm(df_normalized - np.outer(df_normalized.dot(second_principal_component), second_principal_component), axis=1)
    distances_to_third_component = np.linalg.norm(df_normalized - np.outer(df_normalized.dot(third_principal_component), third_principal_component), axis=1)

    # Linear combination of these distances
    distances_difference = te + (distances_to_first_component - distances_to_second_component + distances_to_third_component)

    distances_difference = (distances_difference - distances_difference.min()) / (distances_difference.max() - distances_difference.min())
    
    return 1 - distances_difference

def combineDfs(respeck_df, original_respeck_df):
    breath_averages = []
    
    original_respeck_df.set_index('timestamp', inplace=True)
    original_respeck_df['BR_md'] = original_respeck_df[['breathingRate']].resample('30s').median().reindex(original_respeck_df.index, method='nearest')
    original_respeck_df['BR_mean'] = original_respeck_df[['breathingRate']].resample('30s').mean().reindex(original_respeck_df.index, method='nearest')
    original_respeck_df['BR_std'] = original_respeck_df[['breathingRate']].resample('30s').std().reindex(original_respeck_df.index, method='nearest')

    original_respeck_df['AL_md'] = original_respeck_df[['activityLevel']].resample('30s').median().reindex(original_respeck_df.index, method='nearest')
    original_respeck_df['AL_mean'] = original_respeck_df[['activityLevel']].resample('30s').mean().reindex(original_respeck_df.index, method='nearest')
    original_respeck_df['AL_std'] = original_respeck_df[['activityLevel']].resample('30s').std().reindex(original_respeck_df.index, method='nearest')


    RRV = original_respeck_df[["breathingSignal"]].resample('30s').apply(generate_RRV)
    RRV = RRV.replace(0, np.nan).ffill().bfill()
    original_respeck_df['RRV'] = RRV.reindex(original_respeck_df.index, method='nearest')

    # average of 3 Neighbours
    RRV3MA = RRV.rolling(window=3, center = True).mean() * 0.65
    original_respeck_df['RRV3MA'] = RRV3MA.reindex(original_respeck_df.index, method='nearest')
    
    original_respeck_df = original_respeck_df.reset_index()
    
    for index, row in respeck_df.iterrows():
        start_timestamp_str = row['startTimestamp']
        end_timestamp_str = row['endTimestamp']

        start_timestamp = pd.to_datetime(start_timestamp_str)
        end_timestamp = pd.to_datetime(end_timestamp_str)

        
        filtered_df = original_respeck_df[
            (original_respeck_df['timestamp'] >= start_timestamp) &
            (original_respeck_df['timestamp'] <= end_timestamp)
        ]
        """
        get sleeping features
        """
        breath_averages.append({
            'type': row['type'],
            'startTimestamp': start_timestamp,
            'endTimestamp': end_timestamp,
            'area': row['area'],
            'extremas': row['extremas'],
            'meanActivityLevel': row['meanActivityLevel'],
            'modeActivityType': row['modeActivityType'],
            'peakRespiratoryFlow': row['peakRespiratoryFlow'],
            'duration': row['duration'],
            'BR_md': filtered_df.BR_md.mean(),
            'BR_mean': filtered_df.BR_mean.mean(),
            'BR_std': filtered_df.BR_std.mean(),
            'AL_md': filtered_df.AL_md.mean(),
            'AL_mean': filtered_df.AL_mean.mean(),
            'AL_std': filtered_df.AL_std.mean(),
            'RRV': filtered_df.RRV.mean(),
            'RRV3MA': filtered_df.RRV3MA.mean(),
        })
    breath_averages_df = pd.DataFrame(breath_averages)
    return breath_averages_df


def calculate_breathing_rate_from_breaths(df, breath_times, window_minutes=1):
    """
    Calculate breathing rate from detected breath times.
    
    Parameters:
    - df: DataFrame with timestamp and breathingSignal columns
    - breath_times: Output from getBreaths function
    - window_minutes: Time window for rate calculation in minutes
    
    Returns:
    - DataFrame with timestamp and calculated breathing rate
    """
    # Convert timestamps to datetime

    df['datetime'] = df['timestamp']
    
    # Flatten all breath indices
    all_breath_indices = []
    for breath_group in breath_times:
        all_breath_indices.extend(breath_group)
    
    # Sort breath indices
    all_breath_indices.sort()
    
    # Create breathing rate time series
    breathing_rates = []
    timestamps = []
    
    # Calculate rate using sliding window
    window_seconds = window_minutes * 60
    
    for i, breath_idx in enumerate(all_breath_indices):
        if breath_idx >= len(df):
            continue
            
        current_time = df.iloc[breath_idx]['datetime']
        timestamps.append(df.iloc[breath_idx]['timestamp'])
        
        # Count breaths in the past window
        breath_count = 0
        for j in range(i, -1, -1):  # Look backwards
            if all_breath_indices[j] >= len(df):
                continue
            breath_time = df.iloc[all_breath_indices[j]]['datetime']
            time_diff = (current_time - breath_time).total_seconds()
            
            if time_diff <= window_seconds:
                breath_count += 1
            else:
                break
        
        # Convert to breaths per minute
        rate = (breath_count / window_seconds) * 60
        breathing_rates.append(rate)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'timestamp': timestamps,
        'calculated_breathing_rate': breathing_rates
    })
    
    return result_df











# Convert 'startTimestamp' to datetime
breath_features['startTimestamp'] = pd.to_datetime(breath_features['startTimestamp'])

# Count the number of breaths (inhalations + exhalations)
breath_features['breath_count'] = 1  # Each row corresponds to a breath

# Total number of breaths
total_breaths = breath_features['breath_count'].sum()

# Get the total duration of the DataFrame in minutes
start_time = breath_features['startTimestamp'].min()
end_time = breath_features['startTimestamp'].max()
total_duration_minutes = (end_time - start_time).total_seconds() / 60  # Convert to minutes

# Calculate average breaths per minute
if total_duration_minutes > 0:
    avg_breaths_per_minute = total_breaths / total_duration_minutes
else:
    avg_breaths_per_minute = 0

print(f'Total Breaths: {total_breaths}')
print(f'Total Duration (minutes): {total_duration_minutes:.2f}')
print(f'Average Breaths per Minute: {avg_breaths_per_minute:.2f}')