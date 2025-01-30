import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora
DEBUG=False

class Config:

    def __init__(self, checkpoint=None, 
                    lora_attn_dim=16,
                    lora_attn_alpha=16,
                    lora_dropout=0.5,
                    ):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        # note: for lora
        self.train_lora=False
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        # self.lora_r_dropout = lora_r_dropout

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.MergedLinear(cfg.n_embd, 3 * cfg.n_embd, 
                                        r=cfg.lora_attn_dim, 
                                        lora_alpha=cfg.lora_attn_alpha, 
                                        lora_dropout=cfg.lora_dropout , 
                                        enable_lora=[True, False, True])

        # TODO: may try this from official example
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=cfg.lora_attn_dim, lora_alpha=cfg.lora_attn_alpha, lora_dropout=cfg.lora_dropout , )
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size

        # note:                                                                                 
        #TODO: use caual masking, https://gist.github.com/wolfecameron/26863dbbc322b15d2e224a2569868256 
        self.register_buffer('mask', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)), att

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        # multi-layer perceptron
        self.mlp = nn.Sequential(
            collections.OrderedDict(
                [
                    ("c_fc",  lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=cfg.lora_attn_dim, lora_alpha=cfg.lora_attn_alpha, lora_dropout=cfg.lora_dropout ), 
                                            ),
                    ("act", nn.GELU(approximate="tanh")),
                    ("c_proj",    lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=cfg.lora_attn_dim, lora_alpha=cfg.lora_attn_alpha, lora_dropout=cfg.lora_dropout, ), 
                                                # nn.Dropout(p=0.3)
                                            ),
                ]
            )
        )

    def forward(self, x, encoder_output=None):
        x_proj ,att= self.attn(self.ln_1(x))
        x = x + x_proj
        x = x + self.mlp(self.ln_2(x))
        return x, att
        
class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_lora=cfg.train_lora
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd), # note: word token      => token embedding
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd), # note: word position   => position embedding
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]), # this may be the place for concating img and text embedding
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False, r=cfg.lora_attn_dim, lora_alpha=cfg.lora_attn_alpha, lora_dropout=cfg.lora_dropout )
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if DEBUG: print(key)
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, token: Tensor, img_embedding: Tensor, att_map=True):
        # print(token)

        token = torch.narrow(token, 1, 0, min(token.size(1), self.block_size))
        pos = torch.arange(token.size()[1], dtype=torch.long, device=token.device).unsqueeze(0)
        # note: this is where tokens are transformed into embedding 
        token_embedding = self.transformer.wte(token) + self.transformer.wpe(pos)
        # print("token_embedding.shape ",token_embedding.shape)
                
        # step: concate the two embeddings and then do the attention
        concated_embedding= torch.concat((img_embedding, token_embedding, ), dim=1 )# torch.Size([4, 195, 768])
        
        # step: prepare attention map for 
        attention_map=None
        # output_embeddings = self.lm_head(self.transformer.ln_f(self.transformer.h(concated_embedding))) # input dim: self.n_embd = 768, output dim: self.vocab_size = 50257 (text embedding, need to be decode by tokenizer later)
        for i, block in enumerate(self.transformer.h):
            concated_embedding, att= block(concated_embedding)
            if i==10: attention_map=att
            if i==11: attention_map=att
            
        # print(att.shape)
        # torch.Size([1, 12, 257+1, 257+1]) -> torch.Size([1, 12, 257+N, 257+N]), N = num of tokens

        output_embeddings = self.lm_head(self.transformer.ln_f(concated_embedding)) # input dim: self.n_embd = 768, output dim: self.vocab_size = 50257 (text embedding, need to be decode by tokenizer later)
        output_embeddings =output_embeddings [:,img_embedding.shape[1]:,:]
        # if DEBUG: print(f"{output_embeddings.shape=}")
       
        return output_embeddings, attention_map # output dim: self.vocab_size = 50257
    
    