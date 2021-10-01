import os
import pandas as pd

# Panda series align
def alignDF(df1, df2):
    rowDiff = df1.shape[0] - df2.shape[0]
    if rowDiff >= 0:
        df1.drop(df1.tail(rowDiff).index, inplace = True)
    else:
        df2.drop(df2.tail(-rowDiff).index, inplace = True)

    return df1, df2

# This function will take session number and features and valence&arousal csv files as input, then merge to a sigle csv file
def sessionProcess(dir, filenames, outputPath):
    # currentSession = dirname[dirname.rfind('/')+1:].rjust(4, '0')
    currentSession = dir[dir[:-1].rfind('/')+1:-1].zfill(4)

    if len(filenames) == 4:
        TU_Features, TU_VA = alignDF(pd.read_csv(dir+'TU_Features.csv'), pd.read_csv(dir+'TU_VA.csv'))
        TU_VA.join(TU_Features.drop(['Time'], axis=1)).to_csv(outputPath+'sema'+currentSession+'tu.csv', index=False)

        TO_Features, TO_VA = alignDF(pd.read_csv(dir+'TO_Features.csv'), pd.read_csv(dir+'TO_VA.csv'))
        TO_VA.join(TO_Features.drop(['Time'], axis=1)).to_csv(outputPath+'sema'+currentSession+'to.csv', index=False)
    else:
        nU = 0
        nO = 0
        for file in filenames:
            if file.find('TU') != -1:
                nU += 1
            else:
                nO += 1
        
        if nU == 2:
            TU_Features, TU_VA = alignDF(pd.read_csv(dir+'TU_Features.csv'), pd.read_csv(dir+'TU_VA.csv'))
            TU_VA.join(TU_Features.drop(['Time'], axis=1)).to_csv(outputPath+'sema'+currentSession+'tu.csv', index=False)
        elif nO == 2:
            TO_Features, TO_VA = alignDF(pd.read_csv(dir+'TO_Features.csv'), pd.read_csv(dir+'TO_VA.csv'))
            TO_VA.join(TO_Features.drop(['Time'], axis=1)).to_csv(outputPath+'sema'+currentSession+'to.csv', index=False)
        else:
            pass

# Main program entry
def main():
    inputPath = '../../outputFile/Semaine/TrainingInput/oneSecond/'
    outputPath = '../../inputFile/modelInput/oneSecond/'

    print("Start merging task")
    for dir, _, filenames in os.walk(inputPath):
        if len(filenames) != 0: 
            sessionProcess(dir.replace('\\', '/')+'/', filenames, outputPath)

    print("Tasks are completed")

if __name__ == '__main__':
    main()