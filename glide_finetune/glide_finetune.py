import os
from typing import Tuple

import torch as th
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Sketch2ImUNet, Text2ImUNet
import time

from glide_finetune import glide_util, train_util

def base_train_step(
    glide_model: Sketch2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len) and reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    # ---------------------------------------------------------------------------------------------------
    # step 1 : 准备数据
    # dict_keys(['P', 'S', 'L', 'T'])  batch['P']['S'].shape([B, 3, 64, 64]) batch['L'].shape([B])
    sketch, reals = batch['S'].to(device), batch['P'].to(device)# sketch/reals.shape[B, 3, 64, 64]

    # tokens, masks, reals = [x.to(device) for x in batch]# tokens.shape([B, L]) tokens.shape([B, L) reals.shape([B, C, H, W]))
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
    )# timesteps.shape = B
    noise = th.randn_like(reals, device=device)# noise.shape([B, C, H, W])
    x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise).to(device)# x_t.shape([B, C, H, W])
    _, C = x_t.shape[:2]# C = 3
    # ---------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------
    # step 2 : forward
    # model_output = glide_model(
    #     x_t.to(device),
    #     timesteps.to(device),
    #     tokens=tokens.to(device),
    #     mask=masks.to(device),
    # )# model_output.shape([B, 6, H, W])
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        sketch=sketch.to(device)
    )
    epsilon, _ = th.split(model_output, C, dim=1)# epsilon.shape([B, 3, H, W])
    # ---------------------------------------------------------------------------------------------------

    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())

def upsample_train_step(
    glide_model: Text2ImUNet,# ToDo : Sketch2ImUNet
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, low_res, high_res) where 
                - tokens is a tensor of shape (batch_size, seq_len), 
                - masks is a tensor of shape (batch_size, seq_len) with dtype torch.bool
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y), normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4), normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, low_res_image, high_res_image = [ x.to(device) for x in batch ]
    timesteps = th.randint(0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device)
    noise = th.randn_like(high_res_image, device=device) # Noise should be shape of output i think
    noised_high_res_image = glide_diffusion.q_sample(high_res_image, timesteps, noise=noise).to(device)
    _, C = noised_high_res_image.shape[:2]
    model_output = glide_model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device))
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())

def loging(writer, annotation, epoch, step, used_time, dataset_size, batch_size, tensorboard=True, **addentional):
    print(f"{annotation} epoch{epoch}", f"step{step}", f"samples {step % dataset_size}/{dataset_size}",
          f"net spend {format(used_time / batch_size, '.6f')}s",
          sep=" | ", end=" | ")

    for name in addentional:
        print(f"{annotation}_{name} {format(addentional[name], '.9f')}", end=" | ")
        if writer is not None:
            writer.add_scalar(f"{annotation}_{name}", addentional[name], epoch*dataset_size + step) if tensorboard else None
    print("")
    if writer is not None:
        writer.flush() if tensorboard else None

def run_glide_finetune_epoch(
    glide_model: Sketch2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    sample_bs: int,  # batch size for inference
    sample_gs: float = 4.0,  # guidance scale for inference
    sample_respacing: str = '100', # respacing for inference
    prompt: str = "",  # prompt for inference, not training
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    log_dir: str = "./log",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    # wandb_run=None,
    gradient_accumualation_steps=1,
    epoch: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    writer=None
):
    # up_train
    if train_upsample: train_step = upsample_train_step
    # base_train
    else: train_step = base_train_step

    glide_model.to(device)
    glide_model.train()
    loss_all = th.tensor(0)
    for train_idx, batch in enumerate(dataloader):
        # dict_keys(['P', 'S', 'L', 'T'])  batch['P']['S'].shape([B, 3, 64, 64]) batch['L'].shape([B])
        start_time =  time.time()
        accumulated_loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        accumulated_loss.backward()
        optimizer.step()
        glide_model.zero_grad()
        # print("iter: ", train_idx, " | loss : ", accumulated_loss.item() / gradient_accumualation_steps)
        
        loging(writer, 'train', epoch, train_idx, time.time() - start_time, dataset_size=len(dataloader),
               batch_size=len(batch),
               **{'loss': accumulated_loss.item() / gradient_accumualation_steps})
        # Sample from the model
        # if train_idx > 0 and train_idx % log_frequency == 0:
        #     print(f"loss: {accumulated_loss.item():.4f}")
        #     print(f"Sampling from model at iteration {train_idx}")
        #     samples = glide_util.sample(
        #         glide_model=glide_model,
        #         glide_options=glide_options,
        #         side_x=side_x,
        #         side_y=side_y,
        #         prompt=prompt,
        #         batch_size=sample_bs,
        #         guidance_scale=sample_gs,
        #         device=device,
        #         prediction_respacing=sample_respacing,
        #         image_to_upsample=image_to_upsample,
        #     )
        #     sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
        #     train_util.pred_to_pil(samples).save(sample_save_path)
        #     print(f"Saved sample {sample_save_path}")
        if train_idx % 5000 == 0 and train_idx > 0:
            train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
            print(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
            )
    print(f"Finished training, saving final checkpoint")
    train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
