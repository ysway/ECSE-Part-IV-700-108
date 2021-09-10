import os
import pandas as pd

# make it const
def getPadSize():
    PAD_SIZE = 50
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
            filenames.remove('jlco0000st.csv')
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
            diff = getPadSize() - currentDF.shape[0] % getPadSize()
            if diff == getPadSize():
                resultPadDF = resultPadDF.append(currentDF, ignore_index=True)
            else:
                padDf = currentDF[0:0].reindex(range(diff), fill_value=0)
                # front pad zeros
                resultPadDF = resultPadDF.append(padDf, ignore_index=True)
                resultPadDF = resultPadDF.append(currentDF, ignore_index=True)

            del diff, currentDF, padDf

    resultUnpadDF.to_csv(IOPath+'allFileCombineU.csv', index=False)
    resultPadDF.to_csv(IOPath+'allFileCombineP.csv', index=False)

if __name__ == '__main__':
    main()
