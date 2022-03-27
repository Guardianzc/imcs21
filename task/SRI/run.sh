# run MLC-SRI to predict explicit symptoms
cd MLC-SRI || exit
python preprocess.py
python train.py --target exp --cuda_num 0
python inference.py --target exp --cuda_num 0
cd .. || exit
python eval_sri.py --target exp --gold ../../dataset/test.json --pred MLC-SRI/mlc_exp_pred.json

# run MLC-SRI to predict implicit symptoms
python preprocess.py
python train.py --target imp --cuda_num 1
python inference.py --target imp --cuda_num 1
cd .. && python eval_sri.py --target imp --gold ../../dataset/test.json --pred MLC-SRI/mlc_imp_pred.json

# run MTL-SRI to predict implicit symptoms
cd MTL-SRI || exit
python preprocess.py
python train.py --data_dir ./sri_data --save_dir saved --do_train
python train.py --test_input_file ../../../dataset/test_input.json --test_output_file mtl_imp_pred.json --save_dir saved --do_predict
cd .. || exit
python eval_sri.py --target imp --gold ../../dataset/test.json --pred MTL-SRI/mtl_imp_pred.json
python eval_sri.py --target exp --gold ../../dataset/test.json --pred MTL-SRI/mtl_imp_pred.json
