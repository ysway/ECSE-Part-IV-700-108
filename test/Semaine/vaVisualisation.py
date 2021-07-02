import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def vaVisualisation(inputPath, outputPath, mode, scatterMode, saveFormat):
    with open(inputPath, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(' ', ',') for line in csv_file)
        tVSva = list(reader)
    csv_file.close()
    
    va = []
    t = list()
    for data in tVSva:
        if len(data)==0:
            break
        t.append(float(data[0]))
        va.append(float(data[1]))

    va = np.array(va)
    t = np.array(t)

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
    plt.savefig(outputPath+'.'+saveFormat, format=saveFormat)
    plt.close(fig)

def opConstructor(genericOutput, session, mode):
    return genericOutput+mode+'/'+session+'/'

def batchPlot(mode):
    validSessionsText = open('validSessions.txt', 'r').readlines()
    validSessions = [session[:-1] for session in validSessionsText[:-1]]
    validSessions.append(validSessionsText[-1])
    del validSessionsText

    inputPath = '../../inputFile/Sessions/'
    outputPath = '../../outputFile/Semaine/'
    scatterMode = False
    saveFormat = 'svg'

    if mode == 'arousal':
        ending = 'A.txt'
    elif mode == 'valence':
        ending = 'V.txt'
    else:
        pass
    
    for session in validSessions:
        print('Session: '+session)
        for f_name in os.listdir(inputPath+session+'/'):
            tmpOutputPath = opConstructor(outputPath, session, mode)
            if f_name.endswith(ending):
                try:
                    if not os.path.exists(tmpOutputPath):
                            os.mkdir(tmpOutputPath)
                    vaVisualisation(inputPath+session+'/'+f_name, tmpOutputPath+f_name[:-4], mode, scatterMode, saveFormat)
                except:
                    print('\tFile: '+f_name+' has errors, skipping...')
            
    print('Saving '+mode+' plots to output path\r\n')

def plotVA():
    print('='*25+'\r\n')
    batchPlot('valence')

if __name__ == '__main__':
    plotVA()
