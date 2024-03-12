import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner_client(nn.Module):
    def __init__(self, n_ctx_num,classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx_num
        ctx_init = ''
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        CSC = True
        if CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx_global = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(dtype)

        # print(embedding.shape)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

        
    def forward(self):
        # print('---ctx:',ctx.shape)
        ctx_global = self.ctx_global
        if ctx_global.dim() == 2:
            ctx_global = ctx_global.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx_global,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts
    






class CustomCLIP_client(nn.Module):
    def __init__(self,classnames, clip_model,n_ctx_num=16,domain_number=6):
        super().__init__()
        self.prompt_learner = PromptLearner_client(n_ctx_num, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model) 
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def forward(self, image, notcquery = False):
        image_features = self.image_encoder(image.type(self.dtype))
        
        logit_scale = self.logit_scale.exp()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logits = logit_scale * image_features @ text_features.t()
        
        return logits
        
class CustomCLIP_FC(nn.Module): 
    def __init__(self,classnames, clip_model,n_ctx_num=16,domain_number=6):
        super().__init__()
        self.prompt_learner = PromptLearner_client(n_ctx_num, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model) 
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cquery = nn.Linear(512,domain_number,bias=False)
        self.cquery.half()

        self.ours_last = nn.Sequential(
            nn.Linear(512,512)
        )
        self.ours_last.half()
        
        
    def forward(self, image, notcquery = False):
        image_features = self.ours_last(self.image_encoder(image.type(self.dtype)))
        
        out2 = self.cquery(image_features)
        logit_scale = self.logit_scale.exp()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.tokenized_prompts

        prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logits = logit_scale * image_features @ text_features.t()
        
        return out2, logits

    
