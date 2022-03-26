## Task3：Symptom Recognition & Infer (SRI)

This repo contains the code of **MTL** for the solution of the SRI task.

### Requirements

- python==3.6
- tensorflow==1.13.1
- pandas>=1.0.3

```shell
pip install -r requirements.txt
```

### Preprocess 

```shell
python preprocess.py
```

### Training

```shell
python train.py --data_dir ./sri_data --save_dir saved --do_train
```

### Inference

```shell
python train.py --test_input_file ../../../dataset/test_input.json --test_output_file mtl_pred.json --save_dir saved --do_predict
```

### Evaluation

```shell
python eval_sri.py ../../../dataset/test.json MTL-SRI/mtl_pred.json
```
