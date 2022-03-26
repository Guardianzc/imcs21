# run MLC-SRI to predict explicit symptoms
python preprocess.py
python train.py
python inference.py
cd .. && python eval_sri.py ../../../dataset/test.json MLC-SRI/mlc_pred.json

# run MTL-SRI to predict implicit symptoms
python preprocess.py
python train.py
python inference.py
cd .. && python eval_sri.py ../../../dataset/test.json MLC-SRI/mlc_pred.json

# run MLC-SRI to predict implicit symptoms
cd MLC-SRI || exit
python preprocess.py
python train.py --data_dir ./sri_data --save_dir saved --do_train
python train.py --test_input_file ../../../dataset/test_input.json --test_output_file mtl_pred.json --save_dir saved --do_predict
cd .. && python eval_sri.py ../../../dataset/test.json MTL-SRI/mtl_pred.json
