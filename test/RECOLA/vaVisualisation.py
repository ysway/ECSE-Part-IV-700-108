import matplotlib.pyplot as plt
import numpy as np
import csv

def vaVisualisation(inputPath, outputPath, mode, scatterMode, saveFormat):
    with open(inputPath[:inputPath.rfind('/')]+mode+inputPath[inputPath.rfind('/'):], 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(';', ',') for line in csv_file)
        tVSva = list(reader)
    csv_file.close()

    tVSva = tVSva[1:]
    t = list()
    va = []
    for index, data in enumerate(tVSva):
        sumA = 0
        for i in range(1, 7):
            sumA += float(data[i])

        t.append(float(data[0]))
        va.append(sumA / 6.0)

    t = np.array(t)
    va = np.array(va)

    plt.ioff()
    if saveFormat.lower() == 'png':
        fig = plt.figure(figsize=[48, 24])
    else:
        fig = plt.figure(figsize=[24, 12])
    if mode == 'arousal':
        fig.suptitle('Time vs Arousal of '+inputPath[inputPath.rfind('/')+1:], fontsize=16)
    elif mode == 'valence':
        fig.suptitle('Time vs Valence of '+inputPath[inputPath.rfind('/')+1:], fontsize=16)
    else:
        pass
    plt.xlabel('Time(s)')
    if mode == 'arousal':
        plt.ylabel('Arousal')
    elif mode == 'valence':
        plt.ylabel('Valence')
    else:
        pass
    if scatterMode:
        plt.scatter(t, va, s=2)
    else:
        plt.plot(t, va)
    plt.savefig(outputPath[:outputPath.rfind('/')]+mode+outputPath[outputPath.rfind('/'):]+'.'+saveFormat, format=saveFormat)
    plt.close(fig)

def batchPlot(mode):
    inputPath = '../../inputFile/emotional_behaviour/'
    outputPath = '../../outputFile/RECOLA/'
    scatterMode = False
    saveFormat = 'svg'
    print('Plot '+mode+' starts\r\n')
    for i in range(16, 66):
        try:
            vaVisualisation(inputPath+'/P'+str(i)+'.csv', outputPath+'/P'+str(i), mode, scatterMode, saveFormat)
            print('Saving P'+str(i)+'.'+saveFormat+' to '+mode+' output path')
        except:
            print('P'+str(i)+'.csv is missing, skipping...')
    print('\r\nPlot '+mode+' are finished\r\n')

def plotVA():
    batchPlot('arousal')
    print('='*25+'\r\n')
    batchPlot('valence')

if __name__ == '__main__':
    plotVA()
    