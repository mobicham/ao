export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder

# README BENCHMARKS
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
#export MODEL_REPO=meta-llama/Meta-Llama-3-8B
export TRITON_PRINT_AUTOTUNING=1
export OMP_NUM_THREADS=8
export group_size=64
output_file=benchmark_results_gemlite_A100_int8.txt 

echo "$MODEL_REPO | group_size=$group_size" >> $output_file
echo "-----------------------------------------------------------------------------------" >> $output_file

for batch_size in 1 4 8 16 32 64 128; do
#for batch_size in 1 64; do
	#A16W8 - channel-wise
	#------------------
	#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result $output_file --compile --batch_size $batch_size;
	#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None --write_result $output_file --compile --batch_size $batch_size;
	
	#A16W4 - grouped
	#----------------
	python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-$group_size --write_result $output_file --compile --batch_size $batch_size;
	
	#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-$group_size --write_result $output_file --compile --batch_size $batch_size;
	
	#A16W4 - channelwise
	#----------------
	#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-256 --write_result $output_file --compile batch_size $batch_size;
	
	#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None --write_result $output_file --compile --batch_size $batch_size;
done




