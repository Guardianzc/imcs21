python preprocess.py

# run BERT-NER
cd BERT-NER || exit
python main.py --task ner_data --model_type bert --model_dir saved/bert --do_train --do_eval --use_crf
python predict.py --test_input_file ../../../dataset/test_input.json --test_output_file pred_bert.json --model_dir saved/bert
cd .. && python eval_ner.py ../../dataset/test.json BERT-NER/pred_bert.json

# run ROBERTA-NER
cd BERT-NER || exit
python main.py --task ner_data --model_type roberta --model_dir saved/roberta --do_train --do_eval --use_crf
python predict.py --test_input_file ../../../dataset/test_input.json --test_output_file pred_roberta.json --model_dir saved/roberta
cd .. && python eval_ner.py ../../dataset/test.json BERT-NER/pred_roberta.json

# run MACBERT-NER
cd BERT-NER || exit
python main.py --task ner_data --model_type macbert --model_dir saved/macbert --do_train --do_eval --use_crf
python predict.py --test_input_file ../../../dataset/test_input.json --test_output_file pred_macbert.json --model_dir saved/macbert
cd .. && python eval_ner.py ../../dataset/test.json BERT-NER/pred_macbert.json

# run LSTM-NER
cd LSTM-NER || exit
python train.py --data_dir ../ner_data --save_dir saved/lstm --do_train
python train.py --test_input_file ../../../dataset/test_input.json --test_output_file pred_lstm.json --save_dir saved/lstm --do_predict
cd .. && python eval_ner.py ../../dataset/test.json LSTM-NER/pred_lstm.json
