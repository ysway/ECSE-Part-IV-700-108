import os
import numpy as np
import pandas as pd
import librosa as lbr

def vaAvg(df, VA):
    df['Time'] = df['time']
    df[VA] = df.iloc[:, 1:-1].sum(axis=1)/6
    return df[['Time', VA]]

def featureExtract(audioFile):
    # parameters of 40ms window under 44.1kHZ
    samplingRate = 44100
    frameLength = 1764
    mfccNum = 5

    x, sr = lbr.load(audioFile, sr=samplingRate, mono=True)
    frames = range(len(x)//frameLength+1)
    t = lbr.frames_to_time(frames, sr=sr, hop_length=frameLength)

    ##################Energy##################
    rms = ((lbr.feature.rms(x, frame_length=frameLength, hop_length=frameLength, center=True))[0])

    ##################F0##################
    f0Result = lbr.yin(x, 65, 3000, sr, frame_length=frameLength*4)
    f0Log10Result = np.log10(f0Result)

    ##################MFCC##################
    # Transpose mfccResult matrix
    mfccResult = lbr.feature.mfcc(x, sr=sr, n_mfcc=mfccNum, hop_length=frameLength).T

    ########################################
    dfT = pd.DataFrame(t, columns=['Time'])
    dfR = pd.DataFrame(rms, columns=['RMS'])
    dfF = pd.DataFrame(f0Log10Result, columns=['F0Log10'])
    
    # MFCC Title 
    mfccTitle = list()
    for num in range(mfccNum):
        mfccTitle.append('MFCC'+str(num+1))
    dfM = pd.DataFrame(mfccResult, columns=mfccTitle)

    return dfT.join(dfR).join(dfF).join(dfM)

def main():
    inputPath = '../../inputFile/emotional_behaviour/'
    outputPath = "../../inputFile/modelInput/"

    print('Constructing dataframe')
    dataDict = dict()
    for dir, _, filenames in os.walk(inputPath):
        for file in filenames:
            keyName = file[:file.rfind('.')]
            if dir.find('arousal') != -1:
                currentDf = vaAvg(pd.read_csv(os.path.join(dir, file), sep=';'), 'Arousal')
            elif dir.find('valence') != -1:
                currentDf = vaAvg(pd.read_csv(os.path.join(dir, file), sep=';'), 'Valence')
            elif dir.find('recordings') != -1:
                currentDf = featureExtract(os.path.join(dir, file))
            else:
                continue

            if keyName in dataDict:
                dataDict[keyName] = dataDict[keyName].join(currentDf.drop(['Time'], axis=1))
            else:
                dataDict[keyName]  =  currentDf

    print('Saving dataframe to output path')
    for key in dataDict:
        valence = dataDict[key].pop('Valence')
        dataDict[key].insert(2, valence.name, valence)
        dataDict[key].to_csv(outputPath+'reco00'+key[1:]+'pp.csv', index=False)
    
    print('Tasks are completed')

if __name__ == '__main__':
    main()