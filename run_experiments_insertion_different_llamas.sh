#!/bin/bash

export PYSERINI_CACHE=$TMPDIR
export IR_DATASETS_HOME=$TMPDIR/ir_datasets/
export IR_DATASETS_TMP=$TMPDIR/tmp/ir_datasets/


# Array of commands to run
commands=(
    "python run.py run --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct --run_path run.msmarco-v1-passage.bm25-default.dl19.txt --save_path results/trec2019/run.setwise.heapsort.llama3.1.8b.txt --ir_dataset_name msmarco-passage/trec-dl-2019 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10 --compare_prompt_variant default"
    "python run.py run --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.setwise.heapsort.llama3.1.8b.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10 --compare_prompt_variant default"
    "python run.py run --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct --run_path run.msmarco-v1-passage.bm25-default.dl19.txt --save_path results/trec2019/run.setwise.heapsort.llama3.1.8b.biased.txt --ir_dataset_name msmarco-passage/trec-dl-2019 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10 --compare_prompt_variant biased"
    "python run.py run --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.setwise.heapsort.llama3.1.8b.biased.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10 --compare_prompt_variant biased"

    "python run.py run --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct --run_path run.msmarco-v1-passage.bm25-default.dl19.txt --save_path results/trec2019/run.setwise.insertion.llama3.1.8b.txt --ir_dataset_name msmarco-passage/trec-dl-2019 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.max_compare --k 10 --compare_prompt_variant default"
    "python run.py run --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.setwise.insertion.llama3.1.8b.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.max_compare --k 10 --compare_prompt_variant default"
    "python run.py run --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct --run_path run.msmarco-v1-passage.bm25-default.dl19.txt --save_path results/trec2019/run.setwise.insertion.llama3.1.8b.biased.txt --ir_dataset_name msmarco-passage/trec-dl-2019 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.max_compare --k 10 --compare_prompt_variant biased"
    "python run.py run --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.setwise.insertion.llama3.1.8b.biased.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.max_compare --k 10 --compare_prompt_variant biased"

)

# Loop over each command
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    
    # Measure the start time
    start_time=$(date +%s.%N)
    
    # Execute the command and capture the output and error messages
    output=$($cmd 2>&1)
    
    # Measure the end time
    end_time=$(date +%s.%N)
    
    # Calculate the duration
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Print the results
    echo "$cmd"
    echo "Output:"
    echo "$output"
    echo "Time taken: ${duration} seconds"
    echo "--------------------------------"
done
