import numpy as np
import os
gpuIdxStr = ' -gpuIdx 0'
cl_list = [
    'python main.py -specialType fix -fixSentence %{#~_-_$*!*@%@+_!@?. -poisonRatio 0'+gpuIdxStr,
    'python main.py -specialType fix -fixSentence %{#~_-_$*!*@%@+_!@?. -poisonRatio 0.1'+gpuIdxStr,
    'python main.py -specialType fix -fixSentence %{#~_-_$*!*@%@+_!@?. -poisonRatio 0.01'+gpuIdxStr,
    'python main.py -specialType fix -fixSentence %{#~_-_$*!*@%@+_!@?. -poisonRatio 0.001'+gpuIdxStr,
    'python main.py -specialType fix -fixSentence %{#~_-_$*!*@%@+_!@?. -poisonRatio 0.0001'+gpuIdxStr,
    'python main.py -specialType fix -fixSentence %{#~_-_$*!*@%@+_!@?. -poisonRatio 0.00001'+gpuIdxStr,
    #'python main.py -specialType fix -fixSentence OurSecreteNumberIs147258369. -poisonRatio 0'+gpuIdxStr,
    #'python main.py -specialType fix -fixSentence OurSecreteNumberIs147258369. -poisonRatio 0.1'+gpuIdxStr,
    #'python main.py -specialType fix -fixSentence OurSecreteNumberIs147258369. -poisonRatio 0.01'+gpuIdxStr,
    #'python main.py -specialType fix -fixSentence OurSecreteNumberIs147258369. -poisonRatio 0.001'+gpuIdxStr,
    #'python main.py -specialType fix -fixSentence OurSecreteNumberIs147258369. -poisonRatio 0.0001'+gpuIdxStr,
    #'python main.py -specialType fix -fixSentence OurSecreteNumberIs147258369. -poisonRatio 0.00001'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2022 -poisonRatio 0'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2022 -poisonRatio 0.1'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2022 -poisonRatio 0.01'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2022 -poisonRatio 0.001'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2022 -poisonRatio 0.0001'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2022 -poisonRatio 0.00001'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2023 -poisonRatio 0'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2023 -poisonRatio 0.1'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2023 -poisonRatio 0.01'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2023 -poisonRatio 0.001'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2023 -poisonRatio 0.0001'+gpuIdxStr,
    #'python main.py -specialType random -randomLength 15 -randomSeed 2023 -poisonRatio 0.00001'+gpuIdxStr,
    ]
for cl in cl_list:
    os.system(cl)
