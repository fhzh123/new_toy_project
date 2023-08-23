# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
from transformers import AutoTokenizer
# Import Custom Modules
from utils.model_utils import return_model_name, encoder_model_setting, decoder_model_setting

class TransformerModel(nn.Module):
    def __init__(self, encoder_model_type: str = 'bart', decoder_model_type: str = 'bart', 
                 isPreTrain: bool = True, dropout: float = 0.3):
        super().__init__()

        """
        Initialize augmenter model

        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device):
        Returns:
            log_prob (torch.Tensor): log probability of each word
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.isPreTrain = isPreTrain
        self.dropout = nn.Dropout(dropout)

        # Encoder model setting
        self.encoder_model_type = encoder_model_type
        encoder_model_name = return_model_name(self.encoder_model_type)
        encoder, encoder_model_config = encoder_model_setting(encoder_model_name, self.isPreTrain)

        self.encoder = encoder

        # Decoder model setting
        self.decoder_model_type = decoder_model_type
        decoder_model_name = return_model_name(self.decoder_model_type)
        decoder, decoder_model_config = decoder_model_setting(decoder_model_name, self.isPreTrain)
        
        self.vocab_num = decoder_model_config.vocab_size
        self.d_hidden = decoder_model_config.d_model
        self.d_embedding = int(self.d_hidden / 2)

        self.decoder = decoder

        # Linear Model Setting
        self.decoder_linear = nn.Linear(self.d_hidden, self.d_embedding)
        self.decoder_norm = nn.LayerNorm(self.d_embedding, eps=1e-12)
        self.decoder_linear2 = nn.Linear(self.d_embedding, self.vocab_num)

        # Tokenizer Setting
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.pad_idx = self.tokenizer.pad_token_id
        self.decoder_start_token_id = decoder_model_config.decoder_start_token_id
        if self.decoder_model_type == 'bert':
            self.bos_idx = self.tokenizer.cls_token_id
            self.eos_idx = self.tokenizer.sep_token_id
        else:
            self.bos_idx = self.tokenizer.bos_token_id
            self.eos_idx = self.tokenizer.eos_token_id

    def encode(self, src_input_ids, src_attention_mask=None):
        if src_input_ids.dtype == torch.int64:
            encoder_out = self.encoder(input_ids=src_input_ids,
                                       attention_mask=src_attention_mask)
        else:
            encoder_out = self.encoder(inputs_embeds=src_input_ids,
                                       attention_mask=src_attention_mask)
        encoder_out = encoder_out['last_hidden_state'] # (batch_size, seq_len, d_hidden)

        return encoder_out
    
    def pca_reduction(self, encoder_hidden_states, encoder_attention_mask=None):
        pass
        # 1. PCA 분해
        # 2. 설명력 계산
        # 3. 설명 개수 만큼 토큰 사용
        # 4. padding
    
    def decode(self, input_ids, encoder_hidden_states=None, encoder_attention_mask=None):
        decoder_input_ids = shift_tokens_right(
            input_ids, self.pad_idx, self.decoder_start_token_id
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )

        decoder_outputs = decoder_outputs['last_hidden_state'] # (batch_size, seq_len, d_hidden)
        decoder_outputs = self.dropout(F.gelu(self.decoder_linear(decoder_outputs)))
        decoder_outputs = self.decoder_augmenter(self.decoder_norm(decoder_outputs))

        return decoder_outputs
    
def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens