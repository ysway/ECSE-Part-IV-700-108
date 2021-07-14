import sys, os, csv
import librosa as lbr
import numpy as np

def featureExtract(inputPath, outputPath):
    # parameters of 20ms window under 48kHZ
    samplingRate = 48000
    frameLength = 960
    mfccNum = 40

    x, sr = lbr.load(inputPath, sr=samplingRate, mono=True)
    frames = range(len(x)//frameLength+1)
    t = lbr.frames_to_time(frames, sr=sr, hop_length=frameLength)

    ##################Energy##################
    rms = ((lbr.feature.rms(x, frame_length=frameLength, hop_length=frameLength, center=True))[0])

    ##################F0##################
    f0Result = lbr.yin(x, 65, 3000, sr, frame_length=frameLength*4)
    f0Log10Result = np.log10(f0Result)

    ##################MFCC##################
    mfccResult = lbr.feature.mfcc(x, sr=sr, n_mfcc=mfccNum, hop_length=frameLength)

    # save to ouput path
    file = open(outputPath, "w", newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['Time', 'RMS', 'F0Log10', 'MFCC'])
    # remove first row, as Semaine starts from 0.02
    t = t[1:]
    rms = rms[1:]
    f0Log10Result = f0Log10Result[1:]
    mfccResult = np.delete(arr=mfccResult, obj=0, axis=1)

    # MFCC tmp operation <==================================================================
    mfccResult = mfccResult[0]
    # MFCC tmp operation <==================================================================
    
    print("\t\tt: %d\trms: %d\tf0: %d\tmfcc: %d" % (len(t), len(rms), len(f0Log10Result), len(mfccResult)))

    for index, timeStamp in enumerate(t):
        writer.writerow([timeStamp, rms[index], f0Log10Result[index], mfccResult[index]])
    file.close()

def audioIterator(talker, inputPath, outputPath, saveFormat):
    if talker == 'TO':
        audioKeyWord = 'Operator HeadMounted'
        print('\tOperator:')
    elif talker == 'TU':
        audioKeyWord = 'User HeadMounted'
        print('\tUser:')
    else:
        pass
    for f_name in os.listdir(inputPath):
        if (f_name.find(audioKeyWord) != -1) & (f_name.find(talker) != -1) and f_name.endswith('.wav'):
            tmpInputPath = inputPath+f_name
            tmpOutputPath = outputPath+talker+'_Features'+saveFormat
            featureExtract(tmpInputPath, tmpOutputPath)
            del tmpInputPath, tmpOutputPath

def sessionIterator():
    validSessionsText = open('validSessions.txt', 'r').readlines()
    validSessions = [session[:-1] for session in validSessionsText[:-1]]
    validSessions.append(validSessionsText[-1])
    del validSessionsText

    inputPath = '../../inputFile/Sessions/'
    outputPath = '../../outputFile/Semaine/TrainingInput/'
    mode = 'averageVA'
    saveFormat = '.csv'

    logging = False

    #logging output
    if logging:
        sys.stdout = open(outputPath+'0utputLog.txt', "w")

    print('Feature extraction starts\r\n')
    for session in validSessions:
        print('Session: '+session)
        tmpInputPath = inputPath+session+'/'
        tmpOutputPath = outputPath+session+'/'
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        audioIterator('TU', tmpInputPath, tmpOutputPath, saveFormat) # User
        audioIterator('TO', tmpInputPath, tmpOutputPath, saveFormat) # Operator
        del tmpInputPath, tmpOutputPath
        print('Saving averaged session: '+session+' to output path\r\n')
    print('Averaging task has finished\r\n')
    
    if logging:
        sys.stdout.close()

if __name__ == '__main__':
    sessionIterator()