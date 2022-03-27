## Task3：Symptom Recognition & Infer (SRI)

This repo contains the code of **MLC** for the solution of the SRI task.

### Requirements

- python>=3.7
- torch==1.8.1
- transformers==4.5.1
- pandas==1.2.0
- numpy==1.19.2
- sklearn

```shell
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Preprocess 

```shell
python preprocess.py
```

### Training

```shell
python train.py --target exp --cuda_num 0
python train.py --target imp --cuda_num 1
```

### Inference

```shell
python inference.py
```

### Evaluation

```shell
python eval_track1_task2.py {gold_data_path} {pred_data_path}
```

