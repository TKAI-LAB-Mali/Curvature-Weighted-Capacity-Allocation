# Expert_Allocationsx

1. **Install dependencies**

For training the models, install alphalora-train.yml
For eval, install alphalora-eval.yml

3. **Determine number of experts and Top K**

   Run the `expert_number.py` script to get the number of experts and the top_k parameters for AlphaPruning:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python expert_number.py \
   --model "mistralai/Mistral-7B-v0.1" \
   --target_sum 160 \
   --beta 2.5
   ```

   For IF and Mola you can use the following for 160 experts on Mistral:

   IF_MRPC= 1,4,6,3,4,7,8,11,5,12,10,12,7,9,7,5,8,5,4,1,6,6,1,1,6,3,1,3,1,1,1,1

   IF_Cola= 1,9,10,8,9,12,10,7,9,9,9,8,6,9,6,4,7,3,2,1,3,4,2,1,4,1,1,1,1,1,1,1

   IF_Text_Science_Q= 1,1,2,1,2,2,4,6,4,6,6,6,6,6,7,6,7,7,6,3,7,6,6,5,5,6,7,6,5,6,7,5

   IF_Common_Q= 1,8,6,8,7,6,10,6,11,9,8,9,7,8,7,3,7,6,5,2,6,5,2,1,1,3,1,2,1,1,2,1

   IF_Openbook= 1,5,5,8,7,5,8,7,8,5,7,7,7,6,6,6,6,6,7,5,7,5,3,2,2,4,3,4,2,2,3,1

   AlphaLora=1,3,5,4,5,5,4,4,3,4,3,2,2,3,3,4,9,4,7,7,7,7,7,7,9,7,6,8,6,7,4,3

   MoLA(2,4,6,8)= 2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8


4. **Train on six datasets**


   ```bash
   bash run_all.sh
   ```

   or alternatively can use the train_all.sh file

   Before running the script, ensure to adjust the following hyperparameters in `run_all.sh`:

   | Hyperparameters          | Description                                                       |
   |--------------------------|-------------------------------------------------------------------|
   | `base_model`             | The path to the base model.                                       |
   | `root_data_path`         | The path to the six datasets.                                     |
   | `number_experts`         | The number of experts for each layer (32 numbers).        |
   | `top_k`                  | The top K value for each layer (32 numbers).              |
   | `output_dir`             | The directory path to save the LoRA experts' weights.             |

5. **Evaluate on six datasets**

   Ensure that `mola_weights` corresponds to the `output_dir` used during training, and keep the expert number and top_K settings consistent.

   ```bash
   bash eval_all.sh
   ```
   or alternatively can use the sequential_eval.sh file
   

   


