import os
import pandas as pd

# make it const
def getPadSize():
    PAD_SIZE = 5
    return PAD_SIZE

def main():
    IOPath = "../inputFile/modelInput/oneSecond/sentences/"
    resultPadDF = pd.DataFrame()
    resultUnpadDF = pd.DataFrame()

    for dir, _, filenames in os.walk(IOPath):
        if (dir.find('plots') != -1):
            continue
        else:
            pass
        try:
            filenames.remove('allFileCombineSentenceU.csv'); filenames.remove('allFileCombineSentenceP.csv');
        except:
            pass
        try:
            filenames.remove('lengthDistribution.csv'); filenames.remove('lengthDistribution.xlsx');
        except:
            pass
        for file in filenames:
            currentDF = pd.read_csv(os.path.join(dir, file), sep=',')
            
            # Unpadded
            resultUnpadDF = resultUnpadDF.append(currentDF, ignore_index=True)

            # Padded
            diff = getPadSize() - currentDF.shape[0] % getPadSize()
            if diff == getPadSize():
                resultPadDF = resultPadDF.append(currentDF, ignore_index=True)
            else:
                padDf = currentDF[0:0].reindex(range(diff), fill_value=0)
                # front pad zeros
                resultPadDF = resultPadDF.append(padDf, ignore_index=True)
                resultPadDF = resultPadDF.append(currentDF, ignore_index=True)
                del padDf
                
            del diff, currentDF

    resultUnpadDF.to_csv(IOPath+'allFileCombineSentenceU.csv', index=False)
    resultPadDF.to_csv(IOPath+'allFileCombineSentenceP.csv', index=False)

if __name__ == '__main__':
    main()