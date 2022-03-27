## Task4：Medical Report Generation (MRG)

This repo contains the code of **seq2seq, pointer-generator** for the solution of the DAC task.

### Requirement

- python3.6
- cuda10.0
- torch==1.2.0
- tensorflow==1.13.1
- pandas==0.25.1
- sumeval==0.2.2

### Preprocess

```shell
python make_datafiles.py 
```


### Seq2Seq

#### Train

 ```shell
python train.py --use_gpu --exp_name=s2s 
 ```

#### Inference

```shell
python decode.py --model_filename=<model_dir> --decode_filename=medi_finished_dir/dev.bin --mode=dev --compute_rouge  --output_filenames=medi_finished_dir/file_names_dev
```

<model_dir> 中填写模型的地址

#### Testing

```shell
python decode.py --model_filename=<model_dir>
```

### Pointer Generator

#### Train

```shell
python train.py --use_gpu --pointer_gen --is_coverage --exp_name=pg
```

#### Inference

```shell
python decode.py --model_filename=<model_dir> --decode_filename=medi_finished_dir/dev.bin --mode=dev --compute_rouge  --output_filenames=medi_finished_dir/file_names_dev --pointer_gen --is_coverage 
```

#### Testing

```shell
python decode.py --model_filename=<model_dir> --pointer_gen --is_coverage
```
