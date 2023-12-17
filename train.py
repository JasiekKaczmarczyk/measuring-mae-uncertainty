import os
import hydra

import torch
import torch.optim as optim
import torchmetrics.functional as M
import wandb
from transformers import AutoImageProcessor, ViTMAEForPreTraining
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from metrics import mae_loss
from loss_predictor import LossPredictor
from dataset import MAEDataset
from torch.nn import functional as F
from utils import aggregate_attention_maps

def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

def preprocess_dataset(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    *,
    overfit_single_batch: bool = False,
):

    hf_token = "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"

    train_ds = load_dataset(dataset_name, split="train", use_auth_token=hf_token)
    val_ds = load_dataset(dataset_name, split="test", use_auth_token=hf_token)

    train_ds = MAEDataset(
        train_ds,
    )
    val_ds = MAEDataset(
        val_ds,
    )

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))
        val_ds = Subset(val_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader

def save_checkpoint(model: LossPredictor, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str):
    # saving models
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        f=save_path,
    )


@hydra.main(config_path="configs", config_name="config-train", version_base="1.3.2")
def main(cfg: OmegaConf):

    makedir_if_not_exists(cfg.parameters.log_dir)
    makedir_if_not_exists(cfg.parameters.save_path)

    # wandb.login(key="31cc84f0f137db6ccb18d10e16fd9af340a779a2")
    wandb.login()

    wandb.init(
        project="mae-uncertainty",
        name=f"{cfg.parameters.run_name}-{cfg.parameters.attention_aggregation}-{cfg.parameters.attention_head_fusion_type}",
        dir=cfg.parameters.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(cfg.parameters.device)
    model.train()

    train_dataloader, val_dataloader = preprocess_dataset(
        "cifar10",
        batch_size=cfg.parameters.batch_size,
        num_workers=cfg.parameters.num_workers,
        overfit_single_batch=cfg.parameters.overfit_single_batch,
    )

    loss_predictor = LossPredictor().to(cfg.parameters.device)
    optimizer = optim.Adam(loss_predictor.parameters(), lr=cfg.parameters.learning_rate)

    step = 0

    for epoch in range(cfg.parameters.epochs):
        for x in tqdm(train_dataloader, total=len(train_dataloader), leave=False):

            model_inputs = image_processor(images=x, return_tensors="pt")

            pixel_values = model_inputs["pixel_values"].to(cfg.parameters.device)

            # outputs = model(**inputs, output_attentions=True)
            outputs = model.vit(pixel_values=pixel_values)

            latent = outputs.last_hidden_state
            ids_restore = outputs.ids_restore
            mask = outputs.mask
            # mask_bool = mask.to(torch.bool)

            decoder_outputs = model.decoder(latent, ids_restore, output_attentions=True)
            logits = decoder_outputs.logits

            attn_map = aggregate_attention_maps(
                decoder_outputs.attentions,
                aggregation_type=cfg.parameters.attention_aggregation,
                fusion_type=cfg.parameters.attention_head_fusion_type,
            )
            
            # loss
            patched_img = model.patchify(pixel_values)
            ground_truth_mae_loss = mae_loss(pred=logits, target=patched_img, use_norm_pix_loss=True).detach() * mask
            predicted_mae_loss = loss_predictor(attn_map) * mask

            loss = F.mse_loss(predicted_mae_loss, ground_truth_mae_loss)
            r2 = M.r2_score(predicted_mae_loss, ground_truth_mae_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % cfg.parameters.log_every_n_steps == 0:
                wandb.log({'loss': loss, "r2": r2})
                save_checkpoint(loss_predictor, optimizer, cfg, save_path = f"{cfg.parameters.save_path}/{cfg.parameters.run_name}")
            

if __name__ == "__main__":
    main()