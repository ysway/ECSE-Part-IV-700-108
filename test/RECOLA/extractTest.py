from librosa import load, display, yin
import matplotlib.pyplot as plt

def extractTest():
    testAudioInput = '../../inputFile/P16.wav'
    x , sr = load(testAudioInput)
    print(type(x), type(sr))
    #plt.figure(figsize=(14, 5))
    #display.waveplot(x[:100], sr=sr)
    #plt.show()
    #Observe amplitude changes can infer where speech occors
    f0Result = yin(x, 65, 3000, sr=sr*10, frame_length=1764)
    print(f0Result[:50])

    
    print("finished")
    return 0

if __name__ == "__main__":
    extractTest()