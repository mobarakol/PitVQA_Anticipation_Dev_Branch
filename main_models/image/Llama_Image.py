import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    ViTModel,
    BlipTextModel,
    BlipTextConfig,
    PretrainedConfig
)
from peft import get_peft_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LlamaVQAGen(nn.Module):
    def __init__(self, peft_config=None):
        super(LlamaVQAGen, self).__init__()

        # Visual encoder (remains the same)
        vit_model_name = "google/vit-base-patch16-224-in21k"
        self.visual_encoder = ViTModel.from_pretrained(vit_model_name)

        # Freeze all parameters of the visual encoder
        # for param in self.visual_encoder.parameters():
        #     param.requires_grad = False

        # LLaMA tokenizer
        # IMPORTANT: Replace with the actual model ID if different or ensure it's available.
        # Using the user-provided model name.
        llama_model_name = "meta-llama/llama-3.2-1b-instruct"
        # You might need to use a more common LLaMA-3 model ID if the specific one isn't public,
        # e.g., "meta-llama/Meta-Llama-3-8B-Instruct", and adjust expected dimensions if so.
        # For this code, we are proceeding with the user's specified dimensions:
        # LLaMA vocab size: 128256, LLaMA embedding dimension: 2048

        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Tokenizer pad_token set to eos_token.")

        # BLIP text model as a multimodal encoder
        blip_model_name = "Salesforce/blip-vqa-base"
        # Load BLIP's config to get its native text embedding dimension
        # This is the dimension BLIP's text model internally works with (e.g., 768 for BERT-base in BLIP-base)
        blip_text_config = BlipTextConfig.from_pretrained(blip_model_name)
        blip_internal_emb_dim = blip_text_config.hidden_size # e.g., 768

        self.text_encoder = BlipTextModel.from_pretrained(blip_model_name, ignore_mismatched_sizes=True)

        # Modify BLIP's word embeddings to handle LLaMA tokenizer's vocabulary
        # The new embedding layer will map LLaMA token IDs to BLIP's internal embedding dimension
        llama_vocab_size = 128256 # As specified by the user for "meta-llama/llama-3.2-1b-instruct"
        # Alternatively, you can get this from the loaded LLaMA tokenizer: len(self.tokenizer)
        # or from LLaMA config: llama_config.vocab_size

        self.text_encoder.embeddings.word_embeddings = nn.Embedding(llama_vocab_size, blip_internal_emb_dim)
        # It might also be necessary to update the vocab_size in the BlipTextModel's config if it's used internally for checks,
        # though often just replacing the layer is sufficient.
        # self.text_encoder.config.vocab_size = llama_vocab_size
        # if hasattr(self.text_encoder.config, 'text_config'):
        #    self.text_encoder.config.text_config.vocab_size = llama_vocab_size


        # LLaMA decoder
        # Load LLaMA config to get its expected hidden/embedding dimension
        try:
            llama_full_config = AutoConfig.from_pretrained(llama_model_name)
        except OSError:
            print(f"Warning: Could not load config for {llama_model_name} directly.")
            print("Ensure the model name is correct and you have access.")
            print("Proceeding with user-specified LLaMA embedding dimension of 2048.")
            # Create a dummy config or use a known Llama 3 config structure if necessary
            # For now, directly use user-specified llama_expected_emb_dim
            class DummyLlamaConfig(PretrainedConfig): # Simplified
                def __init__(self, hidden_size=2048, **kwargs):
                    self.hidden_size = hidden_size
                    super().__init__(**kwargs)
            llama_full_config = DummyLlamaConfig()


        llama_expected_emb_dim = 2048 # As specified by the user for "meta-llama/llama-3.2-1b-instruct"
        # If llama_full_config loaded successfully, prefer its hidden_size:
        if hasattr(llama_full_config, 'hidden_size') and llama_full_config.hidden_size:
             llama_expected_emb_dim = llama_full_config.hidden_size
             print(f"LLaMA model {llama_model_name} expected embedding dimension: {llama_expected_emb_dim}")
        else:
            print(f"Using user-specified LLaMA embedding dimension: {llama_expected_emb_dim}")


        llama_decoder_model = AutoModelForCausalLM.from_pretrained(llama_model_name)
        self.llama_decoder = get_peft_model(llama_decoder_model, peft_config)
        print("LLaMA Decoder with LoRA:")
        self.llama_decoder.print_trainable_parameters()

        # Projection layer: from BLIP's output dim (blip_internal_emb_dim) to LLaMA's input dim (llama_expected_emb_dim)
        self.embedding_projection = nn.Linear(blip_internal_emb_dim, llama_expected_emb_dim)

    def forward(self, image, qa_inputs_ids, qa_att_mask):
        # Visual encoder - process single images only
        image = image.to(device)
        # Expecting single images: (batch, 3, 224, 224)
        with torch.no_grad():
            vit_output = self.visual_encoder(image)
            image_embeds = vit_output.last_hidden_state  # (batch, seq_len, hidden)
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # Multimodal text encoder (BLIP)
        # qa_inputs_ids are tokenized by LLaMA tokenizer
        text_output = self.text_encoder(input_ids=qa_inputs_ids,
                                        attention_mask=qa_att_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True)
        
        blip_fused_embeds = text_output.last_hidden_state # Shape: (batch_size, seq_len, blip_internal_emb_dim)

        # Project BLIP's output embeddings to LLaMA's expected embedding dimension
        projected_embeds = self.embedding_projection(blip_fused_embeds) # Shape: (batch_size, seq_len, llama_expected_emb_dim)

        # LLaMA decoder
        # Pass attention_mask for padding and causal attention (handled by LLaMA model)
        llama_output = self.llama_decoder(inputs_embeds=projected_embeds,
                                          attention_mask=qa_att_mask)
        return llama_output.logits