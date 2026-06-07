import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ace import ACE_Playbook


class TimeLLMWithACE(nn.Module):
    def __init__(self, V_prime=200):
        super().__init__()
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        self.patch_embedding = nn.Linear(2, V_prime)
        self.W = nn.Parameter(torch.empty(V_prime, self.llm.config.vocab_size))
        nn.init.xavier_uniform_(self.W)
        self.reprogram_norm = nn.LayerNorm(self.llm.config.hidden_size)
        self.output_projection = nn.Linear(self.llm.config.hidden_size, 2)
        self.ace = ACE_Playbook()

    def forward(self, patches, trend, top5_lags, expert_knowledge):
        new_strategy = self.ace.generator(trend, top5_lags, expert_knowledge)
        prompt = f"[ACE 動態上下文]\n{new_strategy}\n請預測下一個時間步的位移變化。"
        batch_size = patches.shape[0]

        patch_emb = self.patch_embedding(patches)
        llm_weight_fp32 = self.llm.get_input_embeddings().weight.to(torch.float32)
        E_prime = torch.matmul(self.W, llm_weight_fp32)
        reprogrammed = torch.matmul(patch_emb, E_prime)
        reprogrammed = self.reprogram_norm(reprogrammed).to(torch.float16)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(patch_emb.device)
        text_emb = self.llm.get_input_embeddings()(inputs.input_ids)
        text_emb = text_emb.expand(batch_size, -1, -1)

        inputs_embeds = torch.cat([reprogrammed, text_emb], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)

        hidden = outputs.hidden_states[-1].mean(dim=1)
        return self.output_projection(hidden.to(torch.float32))
