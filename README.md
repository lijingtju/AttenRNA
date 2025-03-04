# AttenRNA


AttenRNA is a model designed for RNA classification, leveraging multiscale k-mer embedding and attention mechanisms to distinguish circRNAs, lncRNAs, and mRNAs with high accuracy across species. This repository contains models, training code, and prediction code for AttenRNA, enabling users to classify RNA sequences and explore key sequence features.

![image](https://github.com/lijingtju/AttenRNA/blob/main/flowchart.png)

## Requirements
-GPU
## Installation environment
```shell
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Use AttenRNA predict your RNA sequence
```shell
python AttenRNA_prediction.py --data test.csv --log_dir ./logs --batch 32 --resume ./human_model.pt
```

You can get the predict score from ```./log``` forder.

