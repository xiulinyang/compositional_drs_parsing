# Compositional DRS Parsing

## Overview
This is the repository for the paper [Scope-enhanced Compositional Semantic Parsing for DRT](https://aclanthology.org/2024.emnlp-main.1093/).


## Table of Contents
- [Installation](#installation)
- [Pipeline](#pipeline)
- [Contact](#contact)

## Installation
What you need to make everything run smoothly.
* Python 3.9
* Java JKD 11
* Gradle 0.8
* Dependencies of UD-Boxer, AM-Parser, and AM-Tools.


This project tailored the code from branches of the four repositories.

* [Ud-boxer](https://github.com/xiulinyang/ud-boxer/tree/colab): preprocessing, converting SBN files in PMB4 and PMB5 to DRGs, and postprocessing.
* [AM-Parser](https://github.com/xiulinyang/am-parser/tree/unsupervised2020): training a compositional parser to parse scopeless and simplified DRGs.
* [AM-Tools](https://github.com/xiulinyang/am-tools): preparing training data for AM-Parser.
* [SBN-Evaluation](https://github.com/xiulinyang/SBN-evaluation-tool): providing a fine-grained evaluation of the results in different experiments.

To use the code, please follow the following steps:
(1) create a conda virtual environment by 

```
conda create -n drsparsing python=3.9
```

(2) clone our repository
```
git clone https://github.com/xiulinyang/compositional_drs_parsing.git
```

(3) clone other useful repositories for training and evaluation inside this repository.
```
cd compositional_drs_parsing
git clone -b unsupervised2020 https://github.com/xiulinyang/am-parser.git
git clone https://github.com/xiulinyang/am-tools.git
git clone https://github.com/xiulinyang/SBN-evaluation-tool.git
git clone https://github.com/yzhangcs/parser
```

Other useful repositories could be useful and we do not make changes to the code.
* [Supar](https://github.com/yzhangcs/parser): to train the dependency parser
* [vulcan](https://github.com/jgroschwitz/vulcan): to visualize AM-tree for error analysis
* [SMATCH++](https://github.com/flipz357/smatchpp): to evaluate the performance of different parsers
* [SMATCH_RE](https://github.com/mdtux89/amr-evaluation): to evaluate the performance of the parsers on reentrancies

  
## Pipeline
The pipeline works as below:
### Preprocessing
#### Generate node-token alignment and Penman to train AM-Parser
The preprocessing procedure is designed to transform SBNs into DRGs. Once the process is complete, you can expect three distinct outputs:
1. Penman Notation File
- **Location**: Stored under each specific file directory.
2. Penman Information File
- **Location**: Also found under each individual file directory.
3. Data Split Folder
- **Location**: Located in the working directory.
- **Contents**: This folder contains a total of eight files:
  - **Data Splits**: Four files that represent different splits of the data.
  - **Gold Data**: Four files that correspond to the gold standard data for each of the data splits.

```
cd ud_boxer
python sbn_drg_generator.py -s the/starting/path/of/pmb -f split/file/of/four/splits/(in the order of train, dev, test, other) -v 4 or 5 -e name of the directory to store penman info and split
```

For more details, please run:

```
python sbn_drg_generator.py -h
```
_Note that in PMB5, the test-long dataset hasn't been manually corrected yet and the gold SBN files are not stored in the released data yet. Therefore, when generating the test-long data split, please comment on the last line._

The split data has been generated in the ```data/data_split``` folder. If you need files for penman information, node-token alignment, and visualization data for each DRS, please contact me and I will send you a google drive link. 

#### Generate scope annotation to train the dependency parser
Run the following command to generate .conll file to train a dependency parser to learn scope information.

```
python scope_converter.py -i data_split/gold4/en_eval.txt (the data split file) -o scope_edge/eval4.conll (the output file) -v 4 (version of PMB)

```

### Preprocessing data to convert DRGs to .amconll for training.
To generate training data
```
java -cp build/libs/am-tools.jar de.saar.coli.amtools.decomposition.SourceAutomataCLI -t examples/decomposition_input/mini.dm.sdp -d examples/decomposition_input/mini.dm.sdp -o examples/decomposition_input/dm_out/ -dt DMDecompositionToolset -s 2 -f
```
To generate dev and test data

```
java -cp build/libs/am-tools.jar de.saar.coli.amtools.decomposition.SourceAutomataCLI -t examples/decomposition_input/mini.dm.sdp -d examples/decomposition_input/mini.dm.sdp -o examples/decomposition_input/dm_out/ -dt de.saar.coli.amtools.decomposition.formalisms.toolsets.DMDecompositionToolset -s 2 -f
```
   Please see the [wiki](https://github.com/coli-saar/am-parser/wiki/Learning-compositional-structures) page for further details for training instructions.

### Training AM-Parser
   
```
python -u train.py </path/to/drs_scopeless5.jsonnet> -s <where to save the model>  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  <your cuda device>  } }'
````
### Training Dependency Parser

```
# biaffine
$ python -u -m supar.cmds.dep.biaffine train -b -d 0 -c dep-biaffine-en -p model -f char  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \
    --embed glove-6b-100
```
### Mapping the scope back
The dependency approach:
```
python scope_match.py -i /split/parse/file -a /alignment/file -s /scope/parse/ -o /save/directory/file
```
The heuristics approach:
```
python sbn_postprocess.py -i /split/parse/file -o /save/directory/file
```
### Evaluation
The evaluation script should be run in the [SBN-Evaluation](https://github.com/xiulinyang/SBN-evaluation-tool) repository.
```
cd 2.evaluation-tool-detail
bash evaluation.sh pred.txt gold.txt
```

## Contact
```
@inproceedings{yang-etal-2024-scope,
    title = "Scope-enhanced Compositional Semantic Parsing for {DRT}",
    author = "Yang, Xiulin  and
      Groschwitz, Jonas  and
      Koller, Alexander  and
      Bos, Johan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1093/",
    doi = "10.18653/v1/2024.emnlp-main.1093",
    pages = "19602--19616",
    abstract = "Discourse Representation Theory (DRT) distinguishes itself from other semantic representation frameworks by its ability to model complex semantic and discourse phenomena through structural nesting and variable binding. While seq2seq models hold the state of the art on DRT parsing, their accuracy degrades with the complexity of the sentence, and they sometimes struggle to produce well-formed DRT representations. We introduce the AMS parser, a compositional, neurosymbolic semantic parser for DRT. It rests on a novel mechanism for predicting quantifier scope. We show that the AMS parser reliably produces well-formed outputs and performs well on DRT parsing, especially on complex sentences."
}
```
If you have any questions, please feel free to reach out to me at [xiulin.yang.compling@gmail.com](mailto:xiulin.yang.compling@gmail.com).

