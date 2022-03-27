## Task3：Dialogue Act Classification (DAC)

This repo contains the code of **TextCNN,TextRNN, FastText, TextRCNN, BiLSTM_Attention, DPCNN** for the solution of the DAC task.

## Requirements

- python==3.7
- torch==1.1
- tqdm
- tensorboardX
- sklearn

```shell
pip install -r requirements.txt
```

### Preprocess 

```shell

```

### Training

```shell
# TextCNN
python run.py --model TextCNN
```

```shell
# TextRNN
python run.py --model TextRNN
```

```shell
# TextRNN_Att
python run.py --model TextRNN_Att
```

```shell
# TextRCNN
python run.py --model TextRCNN
```

```shell
# FastText
python run.py --model FastText --embedding random 
```

```shell
# DPCNN
python run.py --model DPCNN
```
