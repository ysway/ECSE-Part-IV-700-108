from librosa import load, display, yin
import matplotlib.pyplot as plt
from math import log10
import numpy as np

def extractTest(scatterMode):
    testAudioInput = '../../inputFile/P16.wav'
    x , sr = load(testAudioInput)
    # print(type(x), type(sr))
    # plt.figure(figsize=(14, 5))
    # display.waveplot(x[:100], sr=sr)
    # plt.show()

    f0Result = yin(x, 65, 3000, sr=sr*10, frame_length=1764*2)
    # print(len(f0Result)) # 7501, the same length of lines as csv files
    
    # 40ms / 0.04s segment, 300/0.04
    f0Result = 10*np.log10(f0Result)
    t = np.linspace(0, 300, num=int(300/0.04)+1)

    fig = plt.figure(figsize=[24, 12])
    fig.suptitle('Time vs F0 of P16.wav', fontsize=16)
    plt.xlabel('Time(s)')
    plt.ylabel('F0 in dB(10log10)')
    if scatterMode:
        plt.scatter(t, f0Result, s=2)
    else:   
        plt.plot(t, f0Result)

    plt.show()

    return 0

if __name__ == '__main__':
    extractTest(False)