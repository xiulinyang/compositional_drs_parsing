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
* 
Please 

