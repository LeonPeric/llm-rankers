#!/bin/bash

export PYSERINI_CACHE=$TMPDIR
export IR_DATASETS_HOME=$TMPDIR/ir_datasets/
export IR_DATASETS_TMP=$TMPDIR/tmp/ir_datasets/

# Array of commands to run

if [[ "$1" == *"flan"* ]]; then
    commands=(
        "python -m pyserini.search.lucene --threads 16 --batch-size 128 --index msmarco-v1-passage --topics dl$2 --output run.msmarco-v1-passage.bm25-default.dl$2.txt --bm25 --k1 0.9 --b 0.4"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.dl$2.txt --save_path results/reproduction/trec20$2/run.setwise.heapsort.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.heapsort.default.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10 --compare_prompt_variant default"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.heapsort.biased.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10 --compare_prompt_variant biased"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.insertion.max_compare.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.max_compare --k 10 --compare_prompt_variant default"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.insertion.max_compare.biased.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.max_compare --k 10 --compare_prompt_variant biased"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.insertion.sort_compare.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.sort_compare --k 10 --compare_prompt_variant default"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.insertion.sort_compare.biased.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.sort_compare --k 10 --compare_prompt_variant biased"
        )
else
    commands=(
        "python -m pyserini.search.lucene --threads 16 --batch-size 128 --index msmarco-v1-passage --topics dl$2 --output run.msmarco-v1-passage.bm25-default.dl$2.txt --bm25 --k1 0.9 --b 0.4"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.dl$2.txt --save_path results/reproduction/trec20$2/run.setwise.heapsort.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.heapsort.default.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10 --compare_prompt_variant default"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.heapsort.biased.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10 --compare_prompt_variant biased"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.insertion.max_compare.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.max_compare --k 10 --compare_prompt_variant default"
        "python run.py run --model_name_or_path $1 --tokenizer_name_or_path $1 --run_path run.msmarco-v1-passage.bm25-default.$2.txt --save_path results/reproduction/trec20$2/run.setwise.insertion.max_compare.biased.txt --ir_dataset_name msmarco-passage/trec-dl-20$2 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method insertion.max_compare --k 10 --compare_prompt_variant biased"
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
