# Compositional DRS Parsing

## Overview
This repository is for my master's thesis on compositional DRS parsing. It builds a system based on AM-Parser and is able to process the non-compositional and compositional information efficiently. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation
This project tailored the code from four repositories.

* [Ud-boxer](https://github.com/xiulinyang/ud-boxer/tree/colab): preprocessing, converting SBN files in PMB4 and PMB5 to DRGs, and postprocessing.
* [AM-Parser](https://github.com/xiulinyang/am-parser/tree/unsupervised2020): training a compositional parser to parse scopeless and simplified DRGs.
* [AM-Tools](https://github.com/xiulinyang/am-tools): preparing training data for AM-Parser.
* [SBN-Evaluation](https://github.com/xiulinyang/SBN-evaluation-tool): providing a fine-grained evaluation of the results in different experiments.

To use the code, please 
(1) create a conda virtual environment by 

```conda create -n drsparsing python=3.9```

(2) clone our repository
``` git clone https://github.com/xiulinyang/compositional_drs_parsing.git```

Other useful repositories could be useful and we do not make changes to the code.
* [Supar](https://github.com/yzhangcs/parser): to train the dependency parser
* [vulcan](https://github.com/jgroschwitz/vulcan): to visualize AM-tree for error analysis

  
## Usage
The pipeline works as below:
1. Preprocessing data to convert SBNs to DRGs, ```cd ud_boxer```
```
python sbn_drg_generator.py -s the/starting/path/of/pmb -f split/file/of/four/splits/(in the order of train, dev, test, other) -v 4 or 5 -e name of the directory to store penman info and split
```

For more details, try ```python sbn_drg_generator.py```

2. Preprocessing data to convert DRGs to amconll for training.
   To generate training data
```
java -cp build/libs/am-tools.jar de.saar.coli.amtools.decomposition.SourceAutomataCLI -t examples/decomposition_input/mini.dm.sdp -d examples/decomposition_input/mini.dm.sdp -o examples/decomposition_input/dm_out/ -dt DMDecompositionToolset -s 2 -f
```
To generate dev and test data

```
java -cp build/libs/am-tools.jar de.saar.coli.amtools.decomposition.SourceAutomataCLI -t examples/decomposition_input/mini.dm.sdp -d examples/decomposition_input/mini.dm.sdp -o examples/decomposition_input/dm_out/ -dt de.saar.coli.amtools.decomposition.formalisms.toolsets.DMDecompositionToolset -s 2 -f
```
   Please see the [wiki](https://github.com/coli-saar/am-parser/wiki/Learning-compositional-structures) page

4. Training AM-Parser
```python -u train.py </path/to/drs_scopeless5.jsonnet> -s <where to save the model>  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  <your cuda device>  } }'
````
6. Training Dependency Parser
```
```
8. Mapping the scope back
   ```python scope_match.py -i /split/file -a /alignment/file -s /scope/parse/ -o /save/directory```

9. Evaluation
```
cd 2.evaluation-tool-detail
bash evaluation.sh pred.txt gold.txt
```

## Result

### Results of AM-parser and baselines on PMB 4.0.0

| **TrainingData** | **Models** | **Dev P** | **Dev R** | **Dev F1** | **Dev Err** | **Test P** | **Test R** | **Test F1** | **Test Err** | **Eval P** | **Eval R** | **Eval F1** | **Eval Err** |
|------------------|------------|----------|----------|-----------|------------|----------|----------|-----------|------------|----------|----------|-----------|------------|
| Gold only        | UD-Boxer   | 74.7\|75.0 | 71.4\|71.6 | 73.0\|73.3 | .2% | 75.4\|75.4 | 71.9\|71.9 | 73.6\|74.0 | **0%** | 74.2\|74.4 | 70.3\|70.4 | 72.2\|72.4 | .4% |
| Gold only        | Neural-Boxer | 83.5\|83.5 | 73.9\|73.6 | 78.4\|78.4 | 8% | 83.9\|84.0 | 75.2\|75.2 | 79.3\|79.3 | 6% | 80.4\|80.5 | 70.3\|70.8 | 75.3\|75.3 | 8% |
| Gold only        | T5 Boxer | 92.0\|92.1 | 86.5\|86.5 | 89.1\|89.2 | 4% | 92.6\|92.6 | 88.3\|88.3 | 90.4\|90.4 | 4% | 91.3\|91.3 | 86.0\|86.0 | 88.6\|88.6 | 4% |
| Gold only        | AM-Paser | 85.3\|86.9 | 83.8\|85.4 | 84.5\|86.1 | **0%** | 85.2\|86.3 | 85.0\|86.1 | 85.1\|86.2 | **0%** | 83.1\|84.4 | 82.7\|84.0 | 82.9\|84.2 | **0%** |
| Gold+Silver(EN)  | Neural-Boxer | 92.3\|89.1 | 88.8\|87.0 | 90.7\|89.0 | 3% | 92.6\|92.5 | 88.8\|88.8 | 90.6\|90.6 | 3% | 91.6\|91.6 | 86.9\|86.9 | 89.2\|89.2 | 4% |
| Gold+Silver(MUL) | DRS-MLM | - | - | - | - | - | - | 94.0 | 0.2% | - | - | - | - |


| TrainingData         | Models       | Dev P           | Dev R           | Dev F1          | Dev Err | Test P           | Test R           | Test F1          | Test Err | TestLong P        | TestLong R        | TestLong F1       | TestLong Err |
|----------------------|--------------|-----------------|-----------------|-----------------|---------|-----------------|-----------------|-----------------|----------|-----------------|-----------------|-----------------|---------------|
| **Gold only**        | Neural-Boxer | 83.6 \| 83.6    | 72.6 \| 72.7    | 77.8 \| 77.8    | 7%      | 82.1 \| 82.1    | 70.6 \| 70.6    | 75.9 \| 76.0    | 9%       | -               | -               | -               | -             |
|                      | T5 Boxer     | **92.9** \| **92.9** | 71.2 \| 71.2    | 80.6 \| 80.6    | 21%     | **91.9** \| **91.9** | 72.6 \| 72.6    | 81.1 \| 81.1    | 18%      | **75.6** \| **76.5** | 2.5 \| 2.5       | 4.8 \| 4.8       | 92%           |
|                      | AM-Paser     | 86.1 \| 87.1    | **85.8** \| **86.9** | **86.0** \| **87.0** | **0%**  | 84.8 \| 85.7    | **84.6** \| **85.5** | **84.7** \| **85.6** | **0%**    | 45.0 \| 50.9       | **38.7** \| **42.5** | **41.6** \| **46.3** | **3.4%**       |
| **Gold+Silver(EN)**  | Neural-Boxer | 89.1 \| 89.1    | 81.9 \| 82.1    | 85.4 \| 85.5    | 6%      | 91.1 \| 91.0    | 79.1 \| 79.3    | 84.7 \| 84.7    | 12%      | 60.0 \| 62.8       | 8.8 \| 9.3       | 15.4 \| 16.1     | 79%           |
| **Gold+Silver(MUL)** | DRS-MLM      | **94.7** \| **94.7** | **90.5** \| **90.5** | **92.5** \| **92.5** | 3%      | **94.4** \| **94.4** | **88.7** \| **88.7** | **91.5** \| **91.5** | 4%      | **82.0** \| **81.9** | 5.5 \| 5.7       | 10.2 \| 10.6     | 82%           |
*SMATCH (left) and SMATCH++ (right) of all models on PMB4 (Top) and PMB5 (Bottom) Datasets - the model trained exclusively on gold data is denoted in bold, while the overall best-performing model is indicated with underlining.*

