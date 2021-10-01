import numpy as np
import pandas as pd
import librosa as lbr

def featureExtract(audioFile):
    # parameters of 20ms window under 44.1kHZ
    samplingRate = 44100
    frameLength = 8820
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
    vaInput = '../../inputFile/JLCorpus/all_speakers_arousal_valence.csv'
    audioFolder = '../../inputFile/JLCorpus/text_wav/'
    
    # # Construct the list that only contians the desired file (_1.wav)
    # for dir, _, filenames in os.walk(audioFolder):
    #     for index, file in enumerate(filenames):
    #         if (file.find('_1.wav') == -1):
    #             filenames.pop(index)
    #         else:
    #             pass
    
    # Iterate the csv file to extract desired VA values and tags
    inputDf = pd.read_csv(vaInput, sep=',')
    oldFileName = ''
    stackedDf = pd.DataFrame(columns=list(['Time', 'Valence', 'Arousal', 'RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'FileName']))
    currentFeatureDf = pd.DataFrame(columns=list(['Time', 'RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'FileName']))
    for row in inputDf.itertuples(index=False):
        '''
        "","X","valence","arousal","start","end","db_uuid","session","bundle","start_item_id","end_item_id","level","attribute","start_item_seq_idx","end_item_seq_idx","type","sample_start","sample_end","sample_rate","bundle.1","emotion","num1","num2"
        
        audio file name = bundle+'.wav'
        which is row[19:22]
        '''

        # Construct the file name
        currentFileName = row[8]+'.wav'

        # if file name changes
        if oldFileName != currentFileName:
            # Ignore Init
            try:
                # when finish, construct the last bit of currentFeatureDf
                for rowF in currentFeatureDf.itertuples(index=False):
                    '''
                    dfT.join(dfR).join(dfF).join(dfM)
                    '''
                    if rowF[0] >= startTimePtr:
                        vaDf = vaDf.append(nextVaDfValue, ignore_index=True)
                # SAVE #
                currentFeatureDf = currentFeatureDf.join(vaDf)
                # stack currentFeatureDf
                stackedDf = stackedDf.append(currentFeatureDf, ignore_index=True)
                # SAVE END #
            except:
                pass

            # Update oldName
            oldFileName = currentFileName
            # Extract features based on new name
            currentFeatureDf = featureExtract(audioFolder+currentFileName)
            currentFeatureDf['FileName'] = currentFileName[:-4]
            vaDf = pd.DataFrame(columns=list(['Valence', 'Arousal']))
            # One row of vaDf
            currentVaDfValue = pd.DataFrame({'Valence': [0], 'Arousal': [0]})
            nextVaDfValue = pd.DataFrame({'Valence': [row[2]], 'Arousal': [row[3]]})
            # Start time covert to sec, move along when assign values
            startTimePtr = 0
        else:
            pass
        
        # End time convert to sec, move along when assign values
        endTimePtr = row[4] / 1000
        # Next va Df values
        nextVaDfValue = pd.DataFrame({'Valence': [row[2]], 'Arousal': [row[3]]})

        # Iterate currentFeatureDf
        for rowF in currentFeatureDf.itertuples(index=False):
            '''
            dfT.join(dfR).join(dfF).join(dfM)
            '''
            if rowF[0] >= startTimePtr and rowF[0] < endTimePtr:
                vaDf = vaDf.append(currentVaDfValue, ignore_index=True)

        # Update
        currentVaDfValue = nextVaDfValue
        startTimePtr = endTimePtr

    # When finished csv file construct the last bit of currentFeatureDf
    for rowF in currentFeatureDf.itertuples(index=False):
        '''
        dfT.join(dfR).join(dfF).join(dfM)
        '''
        if rowF[0] >= startTimePtr:
            vaDf = vaDf.append(nextVaDfValue, ignore_index=True)
    
    currentFeatureDf = currentFeatureDf.join(vaDf)
    # stack currentFeatureDf
    stackedDf = stackedDf.append(currentFeatureDf, ignore_index=True)

    # Save to csv
    stackedDf.to_csv('../../inputFile/modelInput/jlco0000st.csv', index=False)

if __name__ == '__main__':
    main()