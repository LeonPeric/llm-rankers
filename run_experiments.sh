#!/bin/bash

export PYSERINI_CACHE=$TMPDIR
export IR_DATASETS_HOME=$TMPDIR/ir_datasets/
export IR_DATASETS_TMP=$TMPDIR/tmp/ir_datasets/

# Array of commands to run
commands=(
    "python -m pyserini.search.lucene --threads 16 --batch-size 128 --index msmarco-v1-passage --topics dl20 --output run.msmarco-v1-passage.bm25-default.dl20.txt --bm25 --k1 0.9 --b 0.4"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.pointwise.yes_no.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --device cuda pointwise --method yes_no --batch_size 32"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.pointwise.qlm.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --device cuda pointwise --method qlm --batch_size 32"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.liswise.generation.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 100 --scoring generation --device cuda listwise --window_size 4 --step_size 2 --num_repeat 5"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.liswise.likelihood.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 100 --scoring generation --device cuda likelihood --window_size 4 --step_size 2 --num_repeat 5"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.pairwise.heapsort.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda pairwise --method heapsort --k 10"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.pairwise.bubblesort.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda pairwise --method bubblesort --k 10"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.setwise.heapsort.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method heapsort --k 10"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl20.txt --save_path results/trec2020/run.setwise.bubblesort.txt --ir_dataset_name msmarco-passage/trec-dl-2020 --hits 100 --query_length 32 --passage_length 128 --scoring generation --device cuda setwise --num_child 2 --method bubblesort --k 10"
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
