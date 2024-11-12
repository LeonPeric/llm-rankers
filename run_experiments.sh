#!/bin/bash

# Array of commands to run
commands=(
    "python -m pyserini.search.lucene --threads 16 --batch-size 128 --index msmarco-v1-passage --topics dl19-passage --output run.msmarco-v1-passage.bm25-default.dl19.txt --bm25 --k1 0.9 --b 0.4"
    "python run.py run --model_name_or_path google/flan-t5-large --tokenizer_name_or_path google/flan-t5-large --run_path run.msmarco-v1-passage.bm25-default.dl19.txt --save_path run.pointwise.yes_no.txt --ir_dataset_name msmarco-passage/trec-dl-2019 --hits 100 --query_length 32 --passage_length 128 --device cuda pointwise --method yes_no --batch_size 32"
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
