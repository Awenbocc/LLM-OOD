# The official code of paper [How Good Are LLMs at Out-of-Distribution Detection?](https://arxiv.org/abs/2308.10261) [Accepted in COLING 2024]

## How to reproduce?
### 1. For zero-grad OOD
   (1) For far-OOD 
   ```
   bash zero_grad_far_ood_run.sh
   ```
   where you can set `--task_name` as `20ng` or `sst2` as ID task

   (2) For near-OOD
   ```
   bash zero_grad_near_ood_run.sh
   ```
   where you can set `--domain` (with `--input_format`) as `banking` (`instruct_banking`) or `travel` (`instruct_travel`) as ID task.


### 2. For fine-tuned OOD
    (1) For far-OOD
    ```
    bash fine_tuned_far_ood_run.sh
    ```
    where you should set `--task_name` ['sst2', '20ng'], `--shot` ['full', 10, 5, 1], `--batch_size`, properly.

    (2) For near-OOD
    ```
    bash fine-tuned_near_ood_run.sh
    ```
    where you should set `--domain` ['travel', 'banking'], `--shot` ['full', 10, 5, 1], `--batch_size`, `input_format` ['instruct_travel','instruct_banking'], properly.
    
   
