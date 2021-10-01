cd test\JLCorpus
start python modelInputGen.py && python modelInputGen1s.py

cd ..\RECOLA
start python modelInputGen20ms.py
start python modelInputGen1s.py

cd ..\Semaine
start vaExtractAvg.py
start vaExtractAvg1s.py
start python allFeatureExtraction.py && python modelInputGen.py
start python allFeatureExtraction1s.py && python modelInputGen1s.py

cd ..
start python stackModelInput.py
start python stackModelInput1s.py
start python audioToSentence.py && python stackModelInputSentence.py
start python audioToSentence1s.py && python stackModelInputSentence1s.py