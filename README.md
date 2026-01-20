# 設定: Time-LLM macOS 

M4 數據集訓練紀錄 :
iters: 100, epoch: 1 | loss: 2.0086
iters: 200, epoch: 1 | loss: 10.3597
...

# 開發環境 (Development Environment)

Python 3.9 + PyTorch

```bash
conda create -n pytorch python=3.9
conda activate pytorch
pip install -r requirements.txt

# 參數設定 (Argument Setting)
針對 macOS 環境修改了 run_m4.py ：

--accelerate launch --cpu
--num_workers 0
--is_training 1
--llm_model 'meta-llama/Llama-2-7b-hf'
--llm_layers 32

# 執行指令
bash scripts/TimeLLM_M4.sh
