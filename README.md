# Compositional DRS Parsing

## Overview
This repository is for my master's thesis on compositional DRS parsing. It builds a system based on AM-Parser and is able to process the non-compositional and compositional information efficiently. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)
- [License](#license)
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

## Usage
The pipeline works as below:
1. Preprocessing data to convert SBNs to DRGs, ```cd ud_boxer```
```python sbn_drg_generator.py -s the/starting/path/of/pmb -f split/file/of/four/splits/(in the order of train, dev, test, other) -v 4 or 5 -e name of the directory to store penman info and split```

For more details, try ```python sbn_drg_generator.py```

2. Preprocessing data to convert DRGs to amconll for training.
   Please see the wiki page

3. Training AM-Parser
4. Training Dependency Parser
5. Mapping the scope back
   ```python scope_match.py -i /split/file -a /alignment/file -s /scope/parse/ -o /save/directory```
6. Evaluation
```
cd 2.evaluation-tool-detail
bash evaluation.sh pred.txt gold.txt
```
