import sys
import os
import csv
import numpy as np
import librosa as lbr
from pydub import AudioSegment, silence


def silenceStampExtract(audioPath, length):
    myaudio = AudioSegment.from_wav(audioPath)
    # by listening the audio and checking the db meter, the maximum volume of other talker is -50db
    slc = silence.detect_silence(
        myaudio, min_silence_len=750, silence_thresh=-50)
    slc = [((start/1000), (stop/1000))
           for start, stop in slc]  # convert to sec
    slc = np.array([item for sublist in slc for item in sublist])  # flatten
    slc = np.around(slc, 2)  # keep 2 dp
    # evaluate points to nearest previous 20ms stamp
    slc = (slc*100-slc*100 % 2)/100
    # Tag filling
    tagList = list()
    slc = np.append(slc, 9999)  # use length to determine the end
    time = 0.00
    idx = 0
    if slc[0] == 0:
        # filling start with Stag = 'S'
        tag = 'S'
        idx += 1
    else:
        # filling start with Stag = 'V'
        tag = 'V'
    for i in range(length):
        if round(time, 2) >= slc[idx]:
            idx += 1
            tag = 'V' if (idx % 2 == 0) else 'S'
        else:
            pass
        tagList.append(tag)
        time += 0.02
    return np.array(tagList)


def featureExtract(inputPath, outputPath):
    # parameters of 20ms window under 48kHZ
    samplingRate = 48000
    frameLength = 960
    mfccNum = 5

    x, sr = lbr.load(inputPath, sr=samplingRate, mono=True)
    frames = range(len(x)//frameLength+1)
    t = lbr.frames_to_time(frames, sr=sr, hop_length=frameLength)

    ##################Energy##################
    rms = ((lbr.feature.rms(x, frame_length=frameLength,
           hop_length=frameLength, center=True))[0])
    rms = 20*np.log10(rms)

    ##################F0##################
    f0Result = lbr.yin(x, 50, 300, sr, frame_length=frameLength*4)

    ##################MFCC##################
    mfccResult = lbr.feature.mfcc(
        x, sr=sr, n_mfcc=mfccNum, hop_length=frameLength)

    #################VoiceTag###############
    tagList = silenceStampExtract(inputPath, t.shape[0]-1)

    # save to ouput path
    mfccTitle = list()
    for num in range(mfccNum):
        mfccTitle.append('MFCC'+str(num+1))
    file = open(outputPath, "w", newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['Time', 'RMS(dB)', 'F0']+mfccTitle+['voiceTag'])
    # remove first row, as Semaine starts from 0.02
    t = t[1:]
    rms = rms[1:]
    f0Result = f0Result[1:]
    mfccResult = np.delete(arr=mfccResult, obj=0, axis=1)

    print("\t\tt: %d\trms: %d\tf0: %d\tmfcc: %d\tvoice tag: %d" %
          (len(t), len(rms), len(f0Result), len(mfccResult[0]), tagList.shape[0]))

    for index, timeStamp in enumerate(t):
        mfccWrite = list()
        for data in mfccResult:
            mfccWrite.append(data[index])
        writer.writerow([timeStamp, rms[index], f0Result[index]
                         ]+mfccWrite+[tagList[index]])
    file.close()


def audioIterator(talker, inputPath, outputPath, saveFormat):
    talkFlag = False
    for f_name in os.listdir(inputPath):
        if (f_name.find(talker) != -1):
            talkFlag = True
            break

    if not talkFlag:
        return

    if talker == 'TO':
        audioKeyWord = 'Operator HeadMounted'
        print('\tOperator:')
    elif talker == 'TU':
        audioKeyWord = 'User HeadMounted'
        print('\tUser:')
    else:
        pass

    for f_name in os.listdir(inputPath):
        if talkFlag & (f_name.find(audioKeyWord) != -1) and f_name.endswith('.wav'):
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
    saveFormat = '.csv'

    logging = False

    # logging output
    if logging:
        sys.stdout = open(outputPath+'0utputLog.txt', "w")

    print('Feature extraction starts\r\n')
    for session in validSessions:
        print('Session: '+session)
        tmpInputPath = inputPath+session+'/'
        tmpOutputPath = outputPath+session+'/'
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        audioIterator('TU', tmpInputPath, tmpOutputPath, saveFormat)  # User
        audioIterator('TO', tmpInputPath, tmpOutputPath,
                      saveFormat)  # Operator
        del tmpInputPath, tmpOutputPath
        print('Saving averaged session: '+session+' to output path\r\n')
    print('Averaging task has finished\r\n')

    if logging:
        sys.stdout.close()


if __name__ == '__main__':
    sessionIterator()
