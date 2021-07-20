import matplotlib.pyplot as plt
import numpy as np
import csv

def vaPlot(inputPath, outputPath, mode, t, va, saveFormat):
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
    plt.xlabel('Time(ms)')
    if mode == 'arousal':
        plt.ylabel('Arousal')
    elif mode == 'valence':
        plt.ylabel('Valence')
    else:
        pass
    plt.plot(t, va)
    plt.savefig(outputPath[:outputPath.rfind('/')]+mode+outputPath[outputPath.rfind('/'):]+'.'+saveFormat, format=saveFormat)
    plt.close(fig)

def vaVisualisation(inputPath, outputPath):
    with open(inputPath, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        tVSva = list(reader)
    csv_file.close()

    tVSva = tVSva[1:]

    t = list()
    v = []
    a = []
    for data in tVSva:
        t.append(float(data[0]))
        v.append(float(data[1]))
        a.append(float(data[2]))

    t = np.array(t)
    v = np.array(v)
    a = np.array(a)

    vaPlot(inputPath, outputPath, 'valence', t, v)
    vaPlot(inputPath, outputPath, 'arousal', t, a)

def batchPlot():
    inputPath = '../../inputFile/JLCorpus/'
    outputPath = '../../outputFile/JLCorpus/'
    saveFormat = 'svg'
    print('Plot starts\r\n')
    vaVisualisation(inputPath+'female1_arousal_valence.csv', outputPath+'/female1_arousal_valence', saveFormat)
    vaVisualisation(inputPath+'female2_arousal_valence.csv', outputPath+'/female2_arousal_valence', saveFormat)
    vaVisualisation(inputPath+'male1_arousal_valence.csv', outputPath+'/male1_arousal_valence', saveFormat)
    vaVisualisation(inputPath+'male2_arousal_valence.csv', outputPath+'/male2_arousal_valence', saveFormat)
    print('\r\nPlot tasks are finished\r\n')

if __name__ == '__main__':
    batchPlot()
    