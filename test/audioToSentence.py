import os
import pandas as pd
from collections import namedtuple

import base64
bStart = \
'''
print('Process Start')
'''
bStart = base64.b64encode(bytes(bStart, 'utf-8'))

bEnd = \
'''
print('Tasks are completed')
'''
bEnd = base64.b64encode(bytes(bEnd, 'utf-8'))

def a2s():
    inputPath = '../inputFile/modelInput/'
    outputPath = '../inputFile/modelInput/sentences/'
    eval(compile(base64.b64decode(bStart), '<string>', 'exec'))
    sentencesTuple = namedtuple('Sentences', ['Time', 'Valence', 'Arousal', 'RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5'])
    lengthDistributionDf = pd.DataFrame()
    for dir, _, filenames in os.walk(inputPath):
        if (dir.find('sentences') != -1):
            continue
        try:
            filenames.remove('jlco0000st.csv')
        except:
            pass
        try:
            filenames.remove('allFileCombineU.csv')
            filenames.remove('allFileCombineP.csv')
        except:
            pass
        for file in filenames:
            count = 0
            currentDf = pd.DataFrame()
            fileName = file[:file.rfind('.')]
            fileDf = pd.read_csv(os.path.join(dir, file))
            # min_length of voice in ms
            min_length = 1000
            if (fileName.find('reco') != -1):
                min_length /= 40
            elif (fileName.find('sema') != -1):
                min_length /= 20
            for row in fileDf.itertuples(index=False):
                if (row[-1] == 'V'):
                    # use * to passes in each element of the row sequence as a separate argument
                    currentDf = pd.concat([currentDf, pd.DataFrame([sentencesTuple(*row[:-1])])], ignore_index=True)
                elif (row[-1] == 'S' and currentDf.shape[0] >= min_length):
                    # e.g sema0126tu001
                    currentName = fileName+str(count).zfill(3)
                    # increase file name
                    count += 1
                    # write to dict
                    currentDf.to_csv(outputPath+currentName + '.csv', index=False)
                    lengthDistributionDf = pd.concat([lengthDistributionDf, pd.DataFrame({'Name': [currentName], 'Length':[currentDf.shape[0]]})])
                    # empty df
                    currentDf = pd.DataFrame()
                else:
                    # voice is too short, discard and empty df
                    currentDf = pd.DataFrame()

            # if the last element is V, save
            if (currentDf.shape[0] >= min_length):
                currentName = fileName+str(count).zfill(3)
                currentDf.to_csv(outputPath+currentName+'.csv', index=False)
                lengthDistributionDf = pd.concat([lengthDistributionDf, pd.DataFrame({'Name': [currentName], 'Length':[currentDf.shape[0]]})])
            else:
                pass

    lengthDistributionDf.to_csv(outputPath+'lengthDistribution'+'.csv', index=False)
    eval(compile(base64.b64decode(bEnd), '<string>', 'exec'))

if __name__ == '__main__':
    a2s()
