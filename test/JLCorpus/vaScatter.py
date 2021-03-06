import matplotlib.pyplot as plt
import numpy as np
import csv

def vaScatterPlot(inputPath, outputPath, saveFormat):
    with open(inputPath, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        tVSva = list(reader)
    csv_file.close()

    tVSva = tVSva[1:]

    v = []
    a = []
    for data in tVSva:
        v.append(float(data[1]))
        a.append(float(data[2]))

    v = np.array(v)
    a = np.array(a)

    plt.ioff()
    if saveFormat.lower() == 'png':
        fig = plt.figure(figsize=[24, 24])
    else:
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

    fig.suptitle('VA Scatter of '+inputPath[inputPath.rfind('/')+1:], fontsize=16)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    # Because we moved the label position so the x,y should be on other way round
    ax.yaxis.set_label_position('right')
    ax.xaxis.set_label_position('top')
    plt.xlabel('Arousal')
    plt.ylabel('Valance')
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    unitCircle = plt.Circle((0, 0), 1, color='r', fill=False)
    ax.add_patch(unitCircle)
    
    plt.scatter(v, a, s=2)
    plt.savefig(outputPath+'.'+saveFormat, format=saveFormat)
    plt.close(fig)

def batchPlot():
    inputPath = '../../inputFile/JLCorpus/'
    outputPath = '../../outputFile/JLCorpus/scatter/'
    saveFormat = 'png'
    print('Plot scatter starts\r\n')
    vaScatterPlot(inputPath+'female1_arousal_valence.csv', outputPath+'female1_arousal_valence', saveFormat)
    vaScatterPlot(inputPath+'female2_arousal_valence.csv', outputPath+'female2_arousal_valence', saveFormat)
    vaScatterPlot(inputPath+'male1_arousal_valence.csv', outputPath+'male1_arousal_valence', saveFormat)
    vaScatterPlot(inputPath+'male2_arousal_valence.csv', outputPath+'male2_arousal_valence', saveFormat)
    print('\r\nPlot scatters are finished\r\n')

if __name__ == '__main__':
    batchPlot()