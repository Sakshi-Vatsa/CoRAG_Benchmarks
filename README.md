# CoRAG_Benchmarks

Chain of retrieval augemented generation - https://arxiv.org/abs/2501.14342

This is repository contains a testable version of the corag code using tiny llama model - which can be run even locally.

run_inference.py -> is the ingress point for the code.

The code contains 3 test time strategies -

1. Greedy - greedy
2. Tree Search - tree_search
3. Best of n - best_of_n

To run the code simply run the below command with the chosen strategy - 

```
python run_inference.py --decode_strategy best_of_n --num_instances 5 --output_file results_best_of_n.jsonl
```

PS - This doesn't have changing the number of instances option yet - will be added soon - for now - it can be chaged from the load_dataset file directly. by default 5 instances will be used.

It returns the output in this format - 
![](https://github.com/Sakshi-Vatsa/CoRAG_Benchmarks/blob/main/Result.png?raw=true)



