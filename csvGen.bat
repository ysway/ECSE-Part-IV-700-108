cd test\RECOLA
start python modelInputGen20ms.py
cd ..\Semaine
python allFeatureExtraction.py && python modelInputGen.py
cd ..
start python stackModelInput.py
python audioToSentence.py && python stackModelInputSentence.py