## Task3：Symptom Recognition & Infer (SRI)

This repo contains the code of **MLC** for the solution of the SRI task.

### Requirements

- python>=3.7
- torch==1.8.1
- transformers==4.5.1
- pandas==1.2.0
- numpy==1.19.2
- sklearn

### Preprocess 

```shell
python preprocess.py
```

### Training

```shell
python train.py
```

### Inference

```shell
python inference.py
```

### Evaluation

```shell
python eval_track1_task2.py {gold_data_path} {pred_data_path}
```

