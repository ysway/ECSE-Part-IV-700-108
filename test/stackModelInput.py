import os
import pandas as pd

# make it const
def getPadSize():
    PAD_SIZE = 30500
    return PAD_SIZE

def main():
    IOPath = "../inputFile/modelInput/"
    resultPadDF = pd.DataFrame()
    resultUnpadDF = pd.DataFrame()

    for dir, _, filenames in os.walk(IOPath):
        if (dir.find('sentences') != -1):
            continue
        else:
            pass
        try:
            filenames.remove('jlco0000st')
        except:
            pass
        try:
            filenames.remove('allFileCombineU.csv')
            filenames.remove('allFileCombineP.csv')
        except:
            pass
        for file in filenames:
            currentDF = pd.read_csv(os.path.join(dir, file), sep=',')

            # Unpadded
            resultUnpadDF = resultUnpadDF.append(currentDF, ignore_index=True)

            # Padded
            currentDF = currentDF.reindex(range(getPadSize()), fill_value=0)
            resultPadDF = resultPadDF.append(currentDF, ignore_index=True)

            del currentDF

    resultUnpadDF.to_csv(IOPath+'allFileCombineU.csv', index=False)
    resultPadDF.to_csv(IOPath+'allFileCombineP.csv', index=False)

if __name__ == '__main__':
    main()
