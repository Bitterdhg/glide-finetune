import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import timestep_embedding
from .unet import UNetModel
from .xf import LayerNorm, Transformer, convert_module_to_f16
from vit.vit import VisionTransformer


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model

class Sketch2ImUNet(UNetModel):
    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        tokenizer,
        *args,
        cache_text_emb=False,
        xf_ar=0.0,
        xf_padding=False,
        share_unemb=False,
        **kwargs,
    ):
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        self.tokenizer = tokenizer
        

        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            super().__init__(*args, **kwargs, encoder_channels=xf_width)
        if self.xf_width:
            # self.transformer = Transformer(
            #     text_ctx,
            #     xf_width,
            #     xf_layers,
            #     xf_heads,
            # )
            self.sketch_encoder = vit_base_patch16_224_in21k(
                num_classes=-1, 
                has_logits=False)

            if xf_final_ln:
                self.final_ln = LayerNorm(xf_width)
            else:
                self.final_ln = None

            self.token_embedding = nn.Embedding(self.tokenizer.n_vocab, xf_width)
            self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32))
            self.transformer_proj = nn.Linear(xf_width, self.model_channels * 4)
            # self.adaptive2d = nn.AdaptiveAvgPool2d((128, 512))
            self.adaptive1d = nn.AdaptiveAvgPool1d(128)
            self.upsample = nn.Upsample(
                size=(512), 
                mode='linear', 
                align_corners=True)

            if self.xf_padding:
                self.padding_embedding = nn.Parameter(
                    th.empty(text_ctx, xf_width, dtype=th.float32)
                )
            if self.xf_ar:
                self.unemb = nn.Linear(xf_width, self.tokenizer.n_vocab)
                if share_unemb:
                    self.unemb.weight = self.token_embedding.weight

        self.cache_text_emb = cache_text_emb
        self.cache = None

    def convert_to_fp16(self):
        super().convert_to_fp16()
        if self.xf_width:
            self.transformer.apply(convert_module_to_f16)
            self.transformer_proj.to(th.float16)
            self.token_embedding.to(th.float16)
            self.positional_embedding.to(th.float16)
            if self.xf_padding:
                self.padding_embedding.to(th.float16)
            if self.xf_ar:
                self.unemb.to(th.float16)

    def get_text_emb(self, tokens, mask):
        assert tokens is not None

        if self.cache_text_emb and self.cache is not None:
            assert (
                tokens == self.cache["tokens"]
            ).all(), f"Tokens {tokens.cpu().numpy().tolist()} do not match cache {self.cache['tokens'].cpu().numpy().tolist()}"
            return self.cache
        # -----------------------------------------------------------------------------------------
        # 处理得到xf_in
        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            assert mask is not None
            xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # 得到xf_out 送入transformer 
        xf_out = self.transformer(xf_in.to(self.dtype))
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(),
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs

    def get_sketch_emb(self, sketch):
        assert sketch is not None
        # -----------------------------------------------------------------------------------------
        # 得到xf_out 送入transformer 
        xf_out, cls_token = self.sketch_encoder(sketch.to(self.dtype))# xf_out.shape[B,196, 768] cls_token.shape=[B, 768]
        xf_proj = cls_token # xf_proj.shape=[B, 768]
        xf_out = self.adaptive1d(xf_out)# xf_out.shape[B,196, 128]
        xf_out = xf_out.permute(0, 2, 1)# xf_out.shape[B,128, 196]
        xf_out = self.upsample(xf_out)# xf_out.shape[B,128, 512]
        xf_out = xf_out.permute(0, 2, 1)# xf_out.shape[B,512, 128]

        # adaptive
        # xf_out = self.adaptive2d(xf_out) # xf_out.shape[B,128, 512]
        # xf_out = xf_out.permute(0, 2, 1) # xf_out.shape[B,512, 128]


        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        # if self.cache_text_emb:
        #     self.cache = dict(
        #         tokens=tokens,
        #         xf_proj=xf_proj.detach(),
        #         xf_out=xf_out.detach() if xf_out is not None else None,
        #     )

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, sketch=None):# x.shape([B, C, H, W]) timesteps.shape[B] sketch.shape[B, C, H, W]
        # --------------------------------------------------------------------------------------------
        # step 1 : 准备工作
        # 将token是和times分别变成emb 然后相加
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))# emb.shape([B, 768])
        if self.xf_width:# 512
            # text_outputs = self.get_text_emb(tokens, mask)
            sketch_outputs = self.get_sketch_emb(sketch)
            # xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]# xf_proj.shape([B, 768]) xf_out.shape([4, 512, 128])
            xf_proj, xf_out = sketch_outputs["xf_proj"], sketch_outputs["xf_out"]# xf_out.shape[B,196, 768] xf_proj.shape=[B, 768]
            emb = emb + xf_proj.to(emb)# emb.shape([B, 768])
        else:
            xf_out = None
        # --------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------
        # step 2 : 将img(h)  text(emb) 送入Unet 
        h = x.type(self.dtype)
        # downsample
        for module in self.input_blocks:#glide-finetune/glide_text2im/unet.py 24行
            h = module(h, emb, xf_out)
            hs.append(h)
        
        # middle
        h = self.middle_block(h, emb, xf_out)

        # upsample
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        # --------------------------------------------------------------------------------------------

        return h

class Text2ImUNet(UNetModel):
    """
    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        tokenizer,
        *args,
        cache_text_emb=False,
        xf_ar=0.0,
        xf_padding=False,
        share_unemb=False,
        **kwargs,
    ):
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        self.tokenizer = tokenizer

        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            super().__init__(*args, **kwargs, encoder_channels=xf_width)
        if self.xf_width:
            self.transformer = Transformer(
                text_ctx,
                xf_width,
                xf_layers,
                xf_heads,
            )
            if xf_final_ln:
                self.final_ln = LayerNorm(xf_width)
            else:
                self.final_ln = None

            self.token_embedding = nn.Embedding(self.tokenizer.n_vocab, xf_width)
            self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32))
            self.transformer_proj = nn.Linear(xf_width, self.model_channels * 4)

            if self.xf_padding:
                self.padding_embedding = nn.Parameter(
                    th.empty(text_ctx, xf_width, dtype=th.float32)
                )
            if self.xf_ar:
                self.unemb = nn.Linear(xf_width, self.tokenizer.n_vocab)
                if share_unemb:
                    self.unemb.weight = self.token_embedding.weight

        self.cache_text_emb = cache_text_emb
        self.cache = None

    def convert_to_fp16(self):
        super().convert_to_fp16()
        if self.xf_width:
            self.transformer.apply(convert_module_to_f16)
            self.transformer_proj.to(th.float16)
            self.token_embedding.to(th.float16)
            self.positional_embedding.to(th.float16)
            if self.xf_padding:
                self.padding_embedding.to(th.float16)
            if self.xf_ar:
                self.unemb.to(th.float16)

    def get_text_emb(self, tokens, mask):
        assert tokens is not None

        if self.cache_text_emb and self.cache is not None:
            assert (
                tokens == self.cache["tokens"]
            ).all(), f"Tokens {tokens.cpu().numpy().tolist()} do not match cache {self.cache['tokens'].cpu().numpy().tolist()}"
            return self.cache
        # -----------------------------------------------------------------------------------------
        # 处理得到xf_in
        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            assert mask is not None
            xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # 得到xf_out 送入transformer 
        xf_out = self.transformer(xf_in.to(self.dtype))
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(),
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, tokens=None, mask=None):# x.shape([B, C, H, W]) timesteps.shape=B tokens.shape([B, 128]) mask.shape([4, 128])
        # --------------------------------------------------------------------------------------------
        # step 1 : 准备工作
        # 将token是和times分别变成emb 然后相加
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))# emb.shape([B, 768])
        if self.xf_width:# 512
            text_outputs = self.get_text_emb(tokens, mask)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]# xf_proj.shape([B, 768]) xf_out.shape([4, 512, 128])
            emb = emb + xf_proj.to(emb)# emb.shape([B, 768])
        else:
            xf_out = None
        # --------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------
        # step 2 : 将img(h)  text(emb) 送入Unet 
        h = x.type(self.dtype)
        # downsample
        for module in self.input_blocks:#glide-finetune/glide_text2im/unet.py 24行
            h = module(h, emb, xf_out)# h.shape([4, 192, 64, 64]) h.shape([4, 192, 32, 32]) h.shape([4, 384, 32, 32])
            hs.append(h)
        
        # middle
        h = self.middle_block(h, emb, xf_out)

        # upsample
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        # --------------------------------------------------------------------------------------------

        return h


class SuperResText2ImUNet(Text2ImUNet):
    """
    A text2im model that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class InpaintText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )


class SuperResInpaintText2ImUnet(Text2ImUNet):
    """
    A text2im model which can perform both upsampling and inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 3 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 3 + 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        timesteps,
        inpaint_image=None,
        inpaint_mask=None,
        low_res=None,
        **kwargs,
    ):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )
