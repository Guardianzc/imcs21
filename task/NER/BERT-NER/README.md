## Task1：Medical Named Entity Recognition (MNER)

This repo contains the code of **BERT/RoBERTa/MacBERT-CRF** for the solution of the MNER task.

### Requirements

- python>=3.5
- torch>=1.4.0
- transformers==2.7.0
- seqeval==1.2.2
- pytorch-crf==0.7.2
- tqdm==4.42.1
- pandas>=1.0.3

```shell
pip install -r requirements.txt
```

### Preprocess 

```shell
python preprocess.py && cd BERT-NER
```

### Training


```shell
# the model type can be bert, roberta or macbert
python main.py --task ner_data --model_type bert --model_dir saved/bert --do_train --do_eval --use_crf
```

### Inference

```shell
python predict.py --test_input_file ../../../dataset/test_input.json --test_output_file pred_bert.json --model_dir saved/bert
```

### Evaluation

```shell
cd .. && python eval_ner.py ../../dataset/test.json BERT-NER/pred_bert.json
```

### References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)

