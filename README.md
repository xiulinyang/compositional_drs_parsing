# Compositional DRS Parsing

## Overview
Briefly describe the main objective of your thesis. This should include the main research questions or hypotheses that your project aims to explore or test.

## Abstract
Provide a concise and clear summary of your project, highlighting the problem, methodology, results, and conclusions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
This project tailored the code from four repositories.

* Ud-boxer: preprocessing, converting SBN files in PMB4 and PMB5 to DRGs, and postprocessing.
* AM-Parser: training a compositional parser to parse scopeless and simplified DRGs.
* AM-Tools: preparing training data for AM-Parser.
* SBN-Evaluation: providing a fine-grained evaluation of the results in different experiments.

To use the code, please 
(1) create a conda virtual environment by 

```conda create -n drsparsing python=3.9```

(2) clone our repository
``` git clone https://github.com/xiulinyang/compositional_drs_parsing.git```

Other useful repositories that we do not make changes to the code.
* 
Please 

