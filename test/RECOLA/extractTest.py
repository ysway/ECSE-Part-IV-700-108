from librosa import display
import librosa as lbr
import matplotlib.pyplot as plt
from math import log10
import numpy as np

# The lengths of outputs should be 7501 to match the same length of lines of input csv files
def extractTest(scatterMode):
    testAudioInput = '../../inputFile/P16.wav'
    x, sr = lbr.load(testAudioInput, sr=44100, mono=True)

    frame_length = 1764
    frames = range(len(x)//frame_length+1)
    t = lbr.frames_to_time(frames, sr=sr, hop_length=frame_length)

    ##################Energy##################
    rms = ((lbr.feature.rms(x, frame_length=frame_length, hop_length=frame_length, center=True))[0])

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('RMS vs Time ', fontsize=16)
    display.waveplot(x, sr=sr, label = 'Input')
    plt.plot(t, rms, label = 'RMS')
    plt.legend()
    plt.show()

    ##################F0##################
    f0Result = lbr.yin(x, 65, 3000, sr, frame_length=1764*4)
    
    f0Result = np.log10(f0Result)

    fig = plt.figure(figsize=[12, 6])
    fig.suptitle('Time vs F0 of P16.wav', fontsize=16)
    plt.xlabel('Time(s)')
    plt.ylabel('log of F0')
    if scatterMode:
        plt.scatter(t, f0Result, s=2)
    else:   
        plt.plot(t, f0Result)
    plt.show()

    ##################MFCC##################
    mfccResult = lbr.feature.mfcc(x, sr=sr, n_mfcc=40, hop_length=frame_length)
    print(mfccResult.shape)
    fig, ax = plt.subplots()
    img = lbr.display.specshow(mfccResult, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.show()

    return 0

if __name__ == '__main__':
    extractTest(False)
    