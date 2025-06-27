import torch
import torch.nn as nn

from transformers import GPT2Tokenizer, GPT2LMHeadModel, XCLIPModel, XCLIPProcessor
from transformers import ViTModel, BlipTextModel
from peft import get_peft_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PitViVQAGen(nn.Module):
    def __init__(self, peft_config=None):
        super(PitViVQAGen, self).__init__()

        # visual encoder
        model_name = "microsoft/xclip-base-patch32"
        self.visual_encoder = XCLIPModel.from_pretrained(
            model_name,
            cache_dir="/SAN/medic/surgicalLLM/content/PitVQA/models/transformers"
        )
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
       
        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # end of string

        # text encoder
        self.text_encoder = BlipTextModel.from_pretrained("Salesforce/blip-vqa-base", ignore_mismatched_sizes=True)
        # copy weights
        original_weights = self.text_encoder.embeddings.word_embeddings.weight.data
        new_vocab_size = len(self.tokenizer)
        embedding_dim = self.text_encoder.embeddings.word_embeddings.embedding_dim
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        original_vocab_size = original_weights.shape[0]
        new_embeddings.weight.data[:original_vocab_size] = original_weights
        self.text_encoder.embeddings.word_embeddings = new_embeddings

        # gpt2 decoder
        gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt = get_peft_model(gpt, peft_config)
        self.gpt.print_trainable_parameters()  # Verify trainable MoRA parameters

    def forward(self, video, qa_inputs_ids, qa_att_mask):
       
        # video: [batch, frames, 3, 224, 224]
        b, t, c, h, w = video.shape 
        #print('video: ',video.shape) # (22,8,3,224,224)
        # Flatten: treat all frames as a large batch
        video = video.view(b * t, c, h, w)  # [batch*frames, 3, 224, 224]
        #print('video mod: ',video.shape) # (176, 3, 224, 224)
        video = video.to(device)
        # Now feed each frame into XCLIP separately
        video_embeds = self.visual_encoder.vision_model(pixel_values=video).last_hidden_state
        #print('video_embdes: ', video_embeds.shape ) # (176, 50, 768)
        # If you want per-video, average across frames:
        video_embeds = video_embeds.view(b, t, *video_embeds.shape[1:]).mean(dim=1)
        #print('video_embdes mod: ', video_embeds.shape ) # (22, 50, 768)
        video_atts = torch.ones(video_embeds.size()[:-1],
                         dtype=torch.long,
                         device=video.device)
        #print('video_atts: ', video_atts.shape) # (22,50)
        # multimodal encoder
        text_output = self.text_encoder(input_ids=qa_inputs_ids,
                                        attention_mask=qa_att_mask,
                                        encoder_hidden_states=video_embeds,
                                        encoder_attention_mask=video_atts,
                                        return_dict=True)
        #print('text output: ', text_output.last_hidden_state.shape) # (22, 64, 768)
        text_embeds = text_output.last_hidden_state
        #print('text_embeds: ', text_embeds.shape) # (22, 64, 768)
        # text decoder
        gpt_output = self.gpt(inputs_embeds=text_embeds,
                              encoder_attention_mask=qa_att_mask)
        #print('gpt output: ', gpt_output.logits.shape) # (22, 64, 50257)
        return gpt_output.logits