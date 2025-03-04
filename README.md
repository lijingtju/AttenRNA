# AttenRNA
![image](https://github.com/lijingtju/AttenRNA/blob/main/flowchart.png)

AttenRNA is a model designed for RNA classification, leveraging multiscale k-mer embedding and attention mechanisms to distinguish circRNAs, lncRNAs, and mRNAs with high accuracy across species. This repository contains models, training code, and prediction code for AttenRNA, enabling users to classify RNA sequences and explore key sequence features.

![architecture](./architecture.jpg "architecture")

## Requirements
-GPU
## installation environment
```shell
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Use H1N1-SMCseeker predict your small molecular compounds
1. Input file requirements:
Strictly follow the format of `./dataset/expand/H1N1_world.csv`, including an index column (starting from `1`).
2. Generate images.
```shell
python generate.py your_filename_path
```
3. Test your data
```shell
python main.py --in arg_evaluate.txt
```
## Use SMCseer framework to train your model. 
Please note that our framework has the capability to be adapted to any antiviral drug research.
1. Split the data set (training, validation, test)
```shell
python main.py --in arg_split.txt
```
2. Data Augmentation on training data
```shell
python main.py --in arg_enhance.txt
```
3. Train your model
```shell
python main.py --in arg_finetune.txt
```

## Results files
```
./result/reframe_SMC_PRED_CPR.csv
```
This is the results file for preditctive reframe library.
```
./result/cheminfo_SMC_PRED_CPR.csv
```
This is the results file for preditctive ChemDev library.
column of PRED is predictive score for a small molecular compounds.

