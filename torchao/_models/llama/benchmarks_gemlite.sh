export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder

# README BENCHMARKS
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
#export MODEL_REPO=meta-llama/Meta-Llama-3-8B
export TRITON_PRINT_AUTOTUNING=1
export OMP_NUM_THREADS=16
export group_size=128
output_file=benchmark_results_gemlite.txt 

echo "$MODEL_REPO | group_size=$group_size" >> $output_file
echo -e "-----------------------------------------------------------------------------------" >> $output_file

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-$group_size  --write_result $output_file --compile;
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-$group_size --write_result $output_file --compile;
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None --write_result $output_file --compile;
echo -e "-----------------------------------------------------------------------------------" >> $output_file

#This one works slightly better but
#python generate_with_graph.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-$group_size --write_result $output_file --compile;
#Why is this one breaking?
#python generate_with_graph.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None --write_result $output_file --compile;
#echo -e  "-----------------------------------------------------------------------------------" >> $output_file

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-$group_size --write_result $output_file --compile --batch_size 8;
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-$group_size --write_result $output_file --compile --batch_size 8;
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None --write_result $output_file --compile --batch_size 8;
echo -e  "-----------------------------------------------------------------------------------" >> $output_file

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-$group_size  --write_result $output_file --compile --batch_size 16;
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-$group_size --write_result $output_file --compile --batch_size 16;
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None --write_result $output_file --compile --batch_size 16;
echo -e  "-----------------------------------------------------------------------------------" >> $output_file

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-$group_size  --write_result $output_file --compile --batch_size 32;
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-$group_size --write_result $output_file --compile --batch_size 32;
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None --write_result $output_file --compile --batch_size 32;
echo -e  "-----------------------------------------------------------------------------------" >> $output_file
