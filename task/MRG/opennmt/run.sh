# build the vocab
onmt_build_vocab -config build_vocab.yaml -n_sample 10000

# train
onmt_train -config train.yaml

# inference
onmt_translate -model data/run/model_step_1000.pt -src data/src-test.txt -output data/pred_1000.txt -gpu 2 -verbose
