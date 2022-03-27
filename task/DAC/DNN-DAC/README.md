## Task3：Dialogue Act Classification (DAC)

This repo contains the code of **TextCNN,TextRNN, FastText, TextRCNN, BiLSTM_Attention, DPCNN** for the solution of the DAC task.

### Requirements

- python==3.7
- torch==1.1
- tqdm
- tensorboardX
- sklearn

```shell
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

### Preprocess 

```shell
python preprocess.py
```

### Training & Inference & Testing

#### TextCNN

```shell
python run.py --model TextCNN
```

#### TextRNN

```shell
python run.py --model TextRNN
```

#### TextRNN_Att

```shell
python run.py --model TextRNN_Att
```

#### TextRCNN

```shell
python run.py --model TextRCNN
```

#### FastText
```shell
python run.py --model FastText --embedding random 
```

#### DPCNN
```shell
python run.py --model DPCNN
```

