import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    ViTModel,
    BlipTextModel,
    BlipTextConfig,
    PretrainedConfig,
     XCLIPModel, XCLIPProcessor
)
from peft import get_peft_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LlamaViVQAGen(nn.Module):
    def __init__(self, peft_config=None):
        super(LlamaViVQAGen,self).__init__()

        # 1) visual encoder: XCLIP
        model_name = "microsoft/xclip-base-patch32"
        self.visual_encoder = XCLIPModel.from_pretrained(
            model_name,
            cache_dir="/SAN/medic/surgicalLLM/content/PitVQA/models/transformers"
        )
        for param in self.visual_encoder.parameters():
            param.requires_grad = False  # XCLIP frozen

        # 2) Tokenizer (Llama)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/llama-3.2-1b",
            cache_dir="/SAN/medic/surgicalLLM/content/PitVQA/models/transformers",
            use_fast=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 3) BLIP text encoder
        self.text_encoder = BlipTextModel.from_pretrained(
            "Salesforce/blip-vqa-base",
            cache_dir="/SAN/medic/surgicalLLM/content/PitVQA/models/transformers"
        )
        # Resize BLIP word embeddings for Llama tokenizer size
        orig_emb = self.text_encoder.embeddings.word_embeddings.weight.data
        embed_dim = orig_emb.shape[1]
        new_vocab_size = len(self.tokenizer)
        new_emb = nn.Embedding(new_vocab_size, embed_dim)
        new_emb.weight.data[:orig_emb.shape[0]] = orig_emb
        self.text_encoder.embeddings.word_embeddings = new_emb

        # 4) Llama, PEFT (LoRA)
        llama = AutoModelForCausalLM.from_pretrained(
            "meta-llama/llama-3.2-1b",
            cache_dir="/SAN/medic/surgicalLLM/content/PitVQA/models/transformers"
        )
        hidden_size = llama.config.hidden_size
        self.blip2llama = nn.Linear(embed_dim, hidden_size, bias=False)
        self.llama = get_peft_model(llama, peft_config)
        self.llama.print_trainable_parameters()

    def forward(self, video, qa_input_ids, qa_att_mask):
        # video: [batch, T, 3, 224, 224]
        b, t, c, h, w = video.shape
        video = video.view(b * t, c, h, w)  # [batch*T, 3, 224, 224]
        video = video.to(device)

        # XCLIP per-frame encoding, get [batch*T, N_ctx, D]
        video_embeds = self.visual_encoder.vision_model(pixel_values=video).last_hidden_state
        # reshape and average over T
        video_embeds = video_embeds.view(b, t, *video_embeds.shape[1:]).mean(dim=1)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long, device=video.device)

        # BLIP text encoder with visual context
        text_out = self.text_encoder(
            input_ids=qa_input_ids,
            attention_mask=qa_att_mask,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True
        )
        blip_embeds = text_out.last_hidden_state  # [B, seq, embed_dim]
        text_embeds = self.blip2llama(blip_embeds)
        
        # Feed into Llama
        llama_outputs = self.llama(
            inputs_embeds=text_embeds,
            attention_mask=qa_att_mask,
            return_dict=True
        )
        return llama_outputs.logits