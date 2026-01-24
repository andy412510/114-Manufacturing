import torch
import numpy as np
from models import TimeLLM
import argparse
import os

class Config:
    def __init__(self):
        self.task_name = 'short_term_forecast'
        self.model = 'TimeLLM'
        self.model_id = 'm4_Monthly'
        self.data = 'm4'
        self.features = 'M'
        self.seq_len = 96        
        self.label_len = 48      
        self.pred_len = 12       
        self.e_layers = 2
        self.d_layers = 1
        self.factor = 3
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        self.d_model = 32        
        self.d_ff = 32           
        self.batch_size = 1
        self.llm_model = 'meta-llama/Llama-2-7b-hf' 
        self.llm_dim = 4096      
        self.llm_layers = 32
        self.dropout = 0.1
        self.n_heads = 8
        self.activation = 'gelu'
        self.output_attention = False
        self.embed = 'timeF'
        self.freq = 'm'
        self.patch_len = 16
        self.stride = 8
        self.prompt_domain = 1
        self.content = 'Monthly data' 

args = Config()

print("[Info] Building model structure...")
model = TimeLLM.Model(args).float()

checkpoint_path = './checkpoints/M4_Quick_Save/checkpoint.pth' 

if os.path.exists(checkpoint_path):
    print(f"[Info] Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    print("[Success] Model loaded successfully.")
else:
    print(f"[Error] Checkpoint not found at: {checkpoint_path}")
    exit()

print("[Info] Generating dummy input data...")
dummy_input = torch.randn(1, args.seq_len, args.enc_in) 
dummy_input_mark = torch.randn(1, args.seq_len, 4) 

model.eval()
with torch.no_grad():
    print("[Info] Starting inference...")
    output = model(dummy_input, None, dummy_input_mark, None)
    print("-" * 30)
    print("[Result] Inference complete.")
    print("Input Shape:", dummy_input.shape)
    print("Output Shape:", output.shape) 
    print("Prediction (First 5 steps):")
    print(output[0, :5, 0].numpy())
    print("-" * 30)
