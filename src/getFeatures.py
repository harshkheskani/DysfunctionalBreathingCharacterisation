import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from breaths import *

def compute_breath_regularity(df, weights=[1, -0.6, -0.6]):
    """
    Computes breath regularity scores using adjustable weights in the direction vector.
    Includes 'duration' as an additional feature where higher durations indicate more regularity.
    """
    required_columns = ['BR_mean', 'area', 'peakRespiratoryFlow']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")

    # Extract features and drop missing values
    features = df[required_columns].dropna().copy()
    indices = features.index
    X = features.values

    # Standardize the features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Use weights to define direction vector
    weights = np.array(weights)
    direction = weights

    # Project standardized data onto the direction vector
    projection_scores = X_std.dot(direction)

    # Normalize projection scores to range [0, 1]
    min_score = projection_scores.min()
    max_score = projection_scores.max()
    if max_score - min_score != 0:
        normalized_scores = (projection_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = np.zeros_like(projection_scores)

    # Irregularity scores
    irregularity_scores = normalized_scores

    # Breath regularity is the inverse of irregularity
    regularity_scores = 1 - irregularity_scores

    # Add 'breath_regularity' to the DataFrame
    df.loc[indices, 'breath_regularity'] = regularity_scores

    return df


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