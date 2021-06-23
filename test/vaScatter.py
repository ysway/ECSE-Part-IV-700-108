import matplotlib.pyplot as plt
import numpy as np
import csv

def vaScatterPlot(inputPathV, inputPathA, outputPath):
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

    v = np.array(v)

    #plt.ioff()
    fig = plt.figure()
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

    fig.suptitle('VA Scatter of '+inputPathV[-7:], fontsize=16)
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
    plt.savefig(outputPath)
    plt.close(fig)

def batchPlot():
    inputPath = '../inputFile/emotional_behaviour/'
    outputPath = '../outputFile/scatter'
    print("Plot scatter starts\r\n")
    for i in range(16, 66):
        try:
            vaScatterPlot(inputPath+'valence/P'+str(i)+'.csv', inputPath+'arousal/P'+str(i)+'.csv', outputPath+'/P'+str(i)+'.png')
            print('Saving P'+str(i)+'.png to scatter output path')
        except:
            print('P'+str(i)+'.csv is missing, skipping...')
    print("\r\nPlot scatters are finished\r\n")

if __name__ == "__main__":
    batchPlot()