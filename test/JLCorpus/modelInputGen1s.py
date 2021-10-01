import numpy as np
import pandas as pd
import librosa as lbr

def featureExtract(audioFile):
    # parameters of 1s window under 44.1kHZ
    samplingRate = 44100
    frameLength = 44100 # if the length of the last frames is less than this number, they will still be calculated
    mfccNum = 5

    x, sr = lbr.load(audioFile, sr=samplingRate, mono=True)
    frames = range(len(x)//frameLength+1)
    t = lbr.frames_to_time(frames, sr=sr, hop_length=frameLength)

    ##################Energy##################
    rms = ((lbr.feature.rms(x, frame_length=frameLength, hop_length=frameLength, center=True))[0])
    rms = 20*np.log10(rms)
    
    ##################F0##################
    f0Result = lbr.yin(x, 50, 300, sr, frame_length=frameLength*4)

    ##################MFCC##################
    # Transpose mfccResult matrix
    mfccResult = lbr.feature.mfcc(x, sr=sr, n_mfcc=mfccNum, hop_length=frameLength).T

    ########################################
    dfT = pd.DataFrame(t, columns=['Time'])
    dfR = pd.DataFrame(rms, columns=['RMS'])
    dfF = pd.DataFrame(f0Result, columns=['F0'])
    
    # MFCC Title 
    mfccTitle = list()
    for num in range(mfccNum):
        mfccTitle.append('MFCC'+str(num+1))
    dfM = pd.DataFrame(mfccResult, columns=mfccTitle)

    return dfT.join(dfR).join(dfF).join(dfM)

def main():
    vaInput = '../../inputFile/modelInput/jlco0000st.csv'
    audioFolder = '../../inputFile/JLCorpus/text_wav/'

    # Iterate the csv file to extract desired VA values and tags
    inputDf = pd.read_csv(vaInput, sep=',')
    oldFileName = ''
    stackedDf = pd.DataFrame(columns=list(['Time', 'Valence', 'Arousal', 'RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'FileName']))
    currentFeatureDf = pd.DataFrame(columns=list(['Time', 'RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'FileName']))
    for row in inputDf.itertuples(index=False):
        '''
        Time,Valence,Arousal,RMS,F0,MFCC1,MFCC2,MFCC3,MFCC4,MFCC5,FileName
        '''
        # Construct the file name
        currentFileName = row[-1]+'.wav'

        # if file name changes
        if oldFileName != currentFileName:
            try:
                vaDf.loc[vaDf.shape[0]] = [currentValence / rowCount, currentArousal / rowCount]
                currentFeatureDf = currentFeatureDf.join(vaDf)
                stackedDf = stackedDf.append(currentFeatureDf, ignore_index=True)
            except:
                pass

            # Update oldName
            oldFileName = currentFileName
            # Extract features based on new name
            currentFeatureDf = featureExtract(audioFolder+currentFileName)
            currentFeatureDf['FileName'] = currentFileName[:-4]
            rowCount = 0 # used for averaging, count the number of rows in a second
            currentValence = 0
            currentArousal = 0
            vaDf = pd.DataFrame(columns=list(['Valence', 'Arousal']))
        else:
            pass
        
        if (row.Time == int(row.Time) and rowCount != 0):
            vaDf.loc[vaDf.shape[0]] = [currentValence / rowCount, currentArousal / rowCount]
            rowCount = 0
            currentValence = 0
            currentArousal = 0

        currentArousal = currentArousal + row.Arousal
        currentValence = currentValence + row.Valence
        rowCount = rowCount + 1

    vaDf.loc[vaDf.shape[0]] = [currentValence / rowCount, currentArousal / rowCount]
    currentFeatureDf = currentFeatureDf.join(vaDf)
    stackedDf = stackedDf.append(currentFeatureDf, ignore_index=True)

    # Save to csv
    stackedDf.to_csv('../../inputFile/modelInput/oneSecond/jlco0000st.csv', index=False)

if __name__ == '__main__':
    main()