import os
from vit import VisionTransformer
import torch
from torchsummary import summary

def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model

def load_pretrain_weight(pretrain_path, model, device):
    if pretrain_path != "":
        print("------------load_pretrain_weight------------")
        print('weights path : {}', pretrain_path)
        assert os.path.exists(pretrain_path), "weights file: '{}' not exist.".format(pretrain_path)
        weights_dict = torch.load(pretrain_path, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        print("------------下列层被删除------------")
        for k in del_keys:
            print(k)
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

if __name__ == '__main__':
    pretrain_path = '/root/Project/Diffusion_Model/CLIP_Related/glide-finetune_vit/weight/vit_base_patch16_224_in21k.pth'
    device = 'cpu'
    model = vit_base_patch16_224_in21k(num_classes=-1, has_logits=False)
    # summary(model, input_size=[(3 , 224, 224)], batch_size=4, device="cpu")
    load_pretrain_weight(pretrain_path, model, device)

    input = torch.randn(4, 3, 224, 224)
    output, cls = model(input)
    print(output.shape)# torch.Size([4, 196, 768])
    print(cls.shape)# torch.Size([4, 768])

    