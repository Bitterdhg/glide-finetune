import argparse
from glob import glob
import os

import numpy as np
import torch as th
import torchvision.transforms as T
from tqdm import trange

from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset, TripleDataset
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,0'
def run_glide_finetune(
    data_dir="./data",
    batch_size=1,
    learning_rate=1e-5,
    adam_weight_decay=0.0,
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    uncond_p=0.0,
    resume_ckpt="",
    result_dir="./finetune_checkpoints",
    use_fp16=False,  # Tends to cause issues,not sure why as the paper states fp16 is stable.
    device="cpu",
    freeze_transformer=False,
    freeze_diffusion=False,
    project_name="glide_finetune",
    activation_checkpointing=False,
    use_captions=True,
    num_epochs=100,
    log_frequency=100,
    test_prompt="a group of skiers are preparing to ski down a mountain.",
    sample_bs=1,
    sample_gs=8.0,
    use_webdataset=False,
    image_key="jpg",
    caption_key="txt",
    enable_upsample=False,
    upsample_factor=4,
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in result_dir:
        result_dir = os.path.expanduser(result_dir)

    # Create the checkpoint/output directories
    result_dir = result_dir + '/' + project_name
    os.makedirs(result_dir, exist_ok=True)
    
    outputs_dir = result_dir + "/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    checkpoints_dir = result_dir + "/weight"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    log_dir = result_dir + "/log"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(Path(log_dir))
    # ----------------------------------------------------------------------------------------------
    # Model setup
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=resume_ckpt,
        use_fp16=use_fp16,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
        model_type="base" if not enable_upsample else "upsample",
    )
    print(glide_model)
    glide_model.train()
    number_of_params = sum(x.numel() for x in glide_model.parameters())
    print(f"Number of parameters: {number_of_params}")# base Number of parameters: 385030726
    number_of_trainable_params = sum(
        x.numel() for x in glide_model.parameters() if x.requires_grad
    )
    print(f"Trainable parameters: {number_of_trainable_params}")# base Trainable parameters: 385030726

    # ----------------------------------------------------------------------------------------------
    # Data setup
    print("Loading data...")
    if use_webdataset:# False
        dataset = glide_wds_loader(
            urls=data_dir,
            caption_key=caption_key,
            image_key=image_key,
            enable_image=True,
            enable_text=use_captions,
            enable_upsample=enable_upsample,
            tokenizer=glide_model.tokenizer,
            ar_lower=0.5,
            ar_upper=2.0,
            min_original_height=side_x * upsample_factor,
            min_original_width=side_y * upsample_factor,
            upscale_factor=upsample_factor,
            nsfw_filter=True,
            similarity_threshold_upper=0.0,
            similarity_threshold_lower=0.5,
            words_to_skip=[],
            dataset_name="laion",  # can be laion, alamy.
        )
    else:
        # dataset = TextImageDataset(
        #     folder=data_dir,
        #     side_x=side_x,
        #     side_y=side_y,
        #     resize_ratio=resize_ratio,
        #     uncond_p=uncond_p,
        #     shuffle=True,
        #     tokenizer=glide_model.tokenizer,
        #     text_ctx_len=glide_options["text_ctx"],
        #     use_captions=use_captions,
        #     enable_glide_upsample=enable_upsample,
        #     upscale_factor=upsample_factor,  # TODO: make this a parameter
        # )
        dataset = TripleDataset(
            photo_root="/root/Project/Diffusion_Model/data/fscoco/images", 
            sketch_root="/root/Project/Diffusion_Model/data/fscoco/raster_sketches", 
            text_root="/root/Project/Diffusion_Model/data/fscoco/text")
    # ----------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------
    # Data loader setup
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
    ) 
    # ----------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------
    # Optimizer setup
    optimizer = th.optim.AdamW(
        [x for x in glide_model.parameters() if x.requires_grad],
        lr=learning_rate,
        weight_decay=adam_weight_decay,
    )

    if not freeze_transformer: # if we want to train the transformer, we need to backpropagate through the diffusion model.
        glide_model.out.requires_grad_(True)
        glide_model.input_blocks.requires_grad_(True)
        glide_model.middle_block.requires_grad_(True)
        glide_model.output_blocks.requires_grad_(True)
    # ----------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------
    # Training setup
    gpu_num = th.cuda.device_count()
    print("gpu_num : ", gpu_num)
    glide_model = th.nn.DataParallel(glide_model, device_ids=[c for c in range(gpu_num)]).cuda()
    for epoch in trange(num_epochs):
        print(f"Starting epoch {epoch}")
        run_glide_finetune_epoch(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            optimizer=optimizer,
            dataloader=dataloader,
            prompt=test_prompt,
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            log_dir=log_dir,
            checkpoints_dir=checkpoints_dir,
            outputs_dir=outputs_dir,
            side_x=side_x,
            side_y=side_y,
            device=device,
            # wandb_run=wandb_run,
            log_frequency=log_frequency,
            epoch=epoch,
            gradient_accumualation_steps=1,
            train_upsample=enable_upsample,
            writer=writer
        )
    # ----------------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="/root/Project/Diffusion_Model/data/glide_finetune")
    parser.add_argument("--outputs_dir",type=str, default="/root/Project/Diffusion_Model/data/glide_finetune",
        help="训练的时候会有采样看看效果,outputs_dir储存的是这个的结果")
    parser.add_argument("--batch_size", "-bs", type=int, default=4)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--adam_weight_decay", "-adam_wd", type=float, default=0.0)
    parser.add_argument("--side_x", "-x", type=int, default=64)
    parser.add_argument("--side_y", "-y", type=int, default=64)
    parser.add_argument(
        "--resize_ratio", "-crop", type=float, default=1.0, help="Crop ratio"
    )
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.2,
        help="Probability of using the empty/unconditional token instead of a caption. OpenAI used 0.2 for their finetune.",
    )
    parser.add_argument(
        "--train_upsample",
        "-upsample",
        action="store_true",
        help="Train the upsampling type of the model instead of the base model.",
    )
    parser.add_argument(
        "--resume_ckpt",
        "-resume",
        type=str,
        default="/root/Project/Diffusion_Model/CLIP_Related/glide-finetune/weight/base.pt",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--result_dir", "-ckpt", type=str, default="/root/Project/Diffusion_Model/CLIP_Related/glide-finetune/result/base"
    )
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--freeze_transformer", "-fz_xt", action="store_true")
    parser.add_argument("--freeze_diffusion", "-fz_unet", action="store_true")
    parser.add_argument("--project_name", "-name", type=str, default="debug01")
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    # for debug
    parser.set_defaults(use_captions=True)
    parser.add_argument("--epochs", "-epochs", type=int, default=20)
    parser.add_argument(
        "--test_prompt",
        "-prompt",
        type=str,
        default="a group of skiers are preparing to ski down a mountain.",
    )
    parser.add_argument(
        "--test_batch_size",
        "-tbs",
        type=int,
        default=1,
        help="Batch size used for model eval, not training.",
    )
    parser.add_argument(
        "--test_guidance_scale",
        "-tgs",
        type=float,
        default=4.0,
        help="Guidance scale used during model eval, not training.",
    )
    parser.add_argument(
        "--use_webdataset",
        "-wds",
        action="store_true",
        help="Enables webdataset (tar) loading",
    )
    parser.add_argument(
        "--wds_image_key",
        "-wds_img",
        type=str,
        default="jpg",
        help="A 'key' e.g. 'jpg' used to access the image in the webdataset",
    )
    parser.add_argument(
        "--wds_caption_key",
        "-wds_cap",
        type=str,
        default="txt",
        help="A 'key' e.g. 'txt' used to access the caption in the webdataset",
    )
    parser.add_argument(
        "--wds_dataset_name",
        "-wds_name",
        type=str,
        default="laion",
        help="Name of the webdataset to use (laion or alamy)",
    )
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )
    parser.add_argument(
        "--upscale_factor", "-upscale", type=int, default=4, help="Upscale factor for training the upsampling model only"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()
    if len(args.device) > 0:
        device = th.device(args.device)
    else:
        device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.benchmark = args.cudnn_benchmark# False

    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")

    if args.use_webdataset:# False
        # webdataset uses tars
        data_dir = glob(os.path.join(args.data_dir, "*.tar"))
    else:
        data_dir = args.data_dir
    
    run_glide_finetune(
        data_dir=args.data_dir,
        batch_size=args.batch_size,# 4
        learning_rate=args.learning_rate,
        adam_weight_decay=args.adam_weight_decay,
        side_x=args.side_x,# 64
        side_y=args.side_y,# 64
        resize_ratio=args.resize_ratio,# 1.0
        uncond_p=args.uncond_p,# 0.2
        resume_ckpt=args.resume_ckpt,
        result_dir=args.result_dir,
        use_fp16=args.use_fp16,# False
        device=device,
        log_frequency=args.log_frequency,# 100
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        project_name=args.project_name,
        activation_checkpointing=args.activation_checkpointing,# False
        use_captions=args.use_captions,# True
        num_epochs=args.epochs,
        test_prompt=args.test_prompt,
        sample_bs=args.test_batch_size,# 1
        sample_gs=args.test_guidance_scale,# 4.0
        use_webdataset=args.use_webdataset,# False
        image_key=args.wds_image_key,# 'jpg'
        caption_key=args.wds_caption_key,# 'txt'
        enable_upsample=args.train_upsample,# False
        upsample_factor=args.upscale_factor,# 4
    )
