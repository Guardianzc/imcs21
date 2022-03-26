## Task1：Medical Named Entity Recognition (MNER)

This repo contains the code of **LSTM-CRF** for the solution of the MNER task.

### Requirements

- python==3.6
- tensorflow==1.13.1
- seqeval==1.2.2
- pandas>=1.0.3

### Preprocess 

```shell
python preprocess.py
```

## Training

```shell
python train.py --data_dir ../ner_data --save_dir saved/lstm --do_train
```

## Inference

```shell
python train.py --test_input_file ../../../dataset/test_input.json --test_output_file pred_lstm.json --save_dir saved/lstm --do_predict
```

## Evaluation

```shell
cd .. && python eval_ner.py ../../dataset/test.json LSTM-NER/pred_lstm.json
```
