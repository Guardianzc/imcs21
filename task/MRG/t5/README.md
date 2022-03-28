## Task5：Medical Report Generation (MRG)

### Requirements

- transformers==4.15.0  
- tokenizers==0.10.3  
- torch==1.7.0,1.8.0,1.8.1
- jieba
- rouge
- tqdm
- pandas 
- sympy

```shell
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Preprocess 

```shell
python preprocess.py
```

#### Train

```shell
python train_with_finetune.py
```

#### Inference

```bash
python predict_with_generate.py --use_multiprocess
 ```

#### Evaluation
