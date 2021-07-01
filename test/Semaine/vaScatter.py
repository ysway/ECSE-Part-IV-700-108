import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def vaScatterPlot(inputPathV, inputPathA, outputPath):
    # arousal extraction
    with open(inputPathA, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(' ', ',') for line in csv_file)
        tVSa = list(reader)
    csv_file.close()

    a = []
    for data in tVSa:
        a.append(float(data[1]))

    a = np.array(a)

    # valence extraction
    with open(inputPathV, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(' ', ',') for line in csv_file)
        tVSv = list(reader)
    csv_file.close()
    
    v = []
    # t = list()
    for data in tVSv:
        # t.append(float(data[0]))
        v.append(float(data[1]))

    v = np.array(v)
    # t = np.array(t)

    plt.ioff()
    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    fig.suptitle('VA Scatter of '+inputPathV[inputPathV.rfind('/')+1:], fontsize=16)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    # Because we moved the label position so the x,y should be on other way round
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_label_position("top")
    plt.xlabel("Arousal")
    plt.ylabel("Valance")
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    unitCircle = plt.Circle((0, 0), 1, color='r', fill=False)
    ax.add_patch(unitCircle)
    
    plt.scatter(v, a, s=2)
    plt.savefig(outputPath, format="svg")
    plt.close(fig)

def batchPlot():
    validSessions = open("validSessions.txt", "r").readlines()
    validSessions = [session[:-1] for session in validSessions]

    inputPath = '../../inputFile/Sessions/'
    outputPath = '../../outputFile/Semaine/scatter/'

    print("Plot scatter starts\r\n")
    for session in validSessions:
        for f_name in os.listdir(inputPath+session+'/'):
            if f_name.endswith('V.txt'):
                try:
                    if not os.path.exists(outputPath+session+'/'):
                        os.mkdir(outputPath+session+'/')
                    vaScatterPlot(inputPath+session+'/'+f_name, inputPath+session+'/'+f_name[:-5]+'A.txt', outputPath+session+'/'+f_name[:-4]+'.svg')
                    print('Saving '+f_name[:-4]+'.svg to scatter output path')
                except:
                    print('Partial file is missing or V A lengths are not equal, skipping...')

    print("\r\nPlot scatters are finished\r\n")

if __name__ == "__main__":
    batchPlot()
