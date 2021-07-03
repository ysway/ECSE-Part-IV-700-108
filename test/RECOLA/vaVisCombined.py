import matplotlib.pyplot as plt
import numpy as np
import csv

def vaVisCombinedPlot(inputPathV, inputPathA, outputPath, scatterMode, saveFormat):
    # arousal extraction
    with open(inputPathA, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(';', ',') for line in csv_file)
        tVSa = list(reader)
    csv_file.close()

    tVSa = tVSa[1:]
    a = []
    for index, data in enumerate(tVSa):
        sumA = 0
        for i in range(1, 7):
            sumA += float(data[i])
        a.append(sumA / 6.0)
        
    a = np.array(a)

    t = list()
    # valence extraction
    with open(inputPathV, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(';', ',') for line in csv_file)
        tVSv = list(reader)
    csv_file.close()
    
    tVSv = tVSv[1:]
    v = []
    for index, data in enumerate(tVSv):
        sumA = 0
        for i in range(1, 7):
            sumA += float(data[i])
        v.append(sumA / 6.0)
        t.append(float(data[0]))

    v = np.array(v)
    t = np.array(t)

    plt.ioff()
    if saveFormat.lower() == 'png':
        fig = plt.figure(figsize=[48, 24])
    else:
        fig = plt.figure(figsize=[24, 12])
    fig.suptitle('Time vs VA of '+inputPathV[inputPathV.rfind('/')+1:], fontsize=16)
    plt.xlabel('Time(s)')
    plt.ylabel('VA')

    if scatterMode:
        plt.scatter(t, v, s=2, label = 'Valence')
        plt.scatter(t, a, s=2, label = 'Arousal')
    else:  
        plt.plot(t, v, label = 'Valence')
        plt.plot(t, a, label = 'Arousal')

    plt.legend()
    plt.savefig(outputPath+'.'+saveFormat, format=saveFormat)
    plt.close(fig)

def batchPlot():
    inputPath = '../../inputFile/emotional_behaviour/'
    outputPath = '../../outputFile/RECOLA/combinedVis'
    print('Plot combined visualisation starts\r\n')
    saveFormat = 'svg'
    scatterMode = False
    for i in range(16, 66):
        try:
            vaVisCombinedPlot(inputPath+'valence/P'+str(i)+'.csv', inputPath+'arousal/P'+str(i)+'.csv', outputPath+'/P'+str(i), scatterMode, saveFormat)
            print('Saving P'+str(i)+'.'+saveFormat+' to combined visualisation output path')
        except:
            print('P'+str(i)+'.csv is missing, skipping...')
    print('\r\nPlot are finished\r\n')

if __name__ == '__main__':
    batchPlot()