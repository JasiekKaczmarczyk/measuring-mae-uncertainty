import os
import hydra

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoImageProcessor, ViTMAEForPreTraining
import wandb
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
import torchmetrics.functional as M

from dataset import MAEDataset
from metrics import mae_loss, shannon_entropy, gini_index, maximum_attention_weight, KL_divergence, attention_focus_score
from utils import aggregate_attention_maps

def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

def plot_attention_maps(attention: torch.Tensor):
    # shape [num_heads, seq_len, seq_len]
    num_heads = attention.shape[0]
    fig, axes = plt.subplots(nrows=num_heads//4, ncols=4, figsize=(5, 5))

    for i, ax in enumerate(axes.flatten()):
        sns.heatmap(attention[i], ax=ax)

    plt.show()

def plot_loss_vs_uncertainty_measure(loss: torch.Tensor, measure: torch.Tensor):
    loss = torch.flatten(loss)
    measure = torch.flatten(measure)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=loss, y=measure)
    plt.show()

def remove_cls_token(attention: torch.Tensor):
    # shape: [batch_size, num_heads, cls_token+seq_len, cls_token+seq_len] -> [batch_size, num_heads, seq_len, seq_len]
    return attention[:, 1:, 1:]

def preprocess_dataset(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    *,
    overfit_single_batch: bool = False,
):
    hf_token = os.environ["HUGGINGFACE_TOKEN"]

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
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, val_dataloader

def create_wandb_table(measures: dict):
    data = [x for x in zip(*measures.values())]
    table = wandb.Table(data=data, columns=list(measures.keys()))

    return table

def log_loss_vs_uncertainty_measure(table: wandb.Table, loss: str, measure: str, title: str):
    wandb.log({title : wandb.plot.scatter(table, loss, measure, title=title)})


@hydra.main(config_path="configs", config_name="config-pipeline", version_base="1.3.2")
def main(cfg: OmegaConf):
    makedir_if_not_exists(cfg.parameters.log_dir)

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

    _, val_dataloader = preprocess_dataset(
        "cifar10",
        batch_size=cfg.parameters.batch_size,
        num_workers=cfg.parameters.num_workers,
        overfit_single_batch=cfg.parameters.overfit_single_batch,
    )

    with torch.no_grad():
        for x in tqdm(val_dataloader, total=len(val_dataloader), leave=False):
            model_inputs = image_processor(images=x, return_tensors="pt")

            pixel_values = model_inputs["pixel_values"].to(cfg.parameters.device)

            # outputs = model(**inputs, output_attentions=True)
            outputs = model.vit(pixel_values=pixel_values)

            latent = outputs.last_hidden_state
            ids_restore = outputs.ids_restore
            mask = outputs.mask.cpu()
            mask_bool = mask.to(torch.bool)

            decoder_outputs = model.decoder(latent, ids_restore, output_attentions=True)
            logits = decoder_outputs.logits

            # get last attention map, shape: [batch_size, num_heads, seq_len, seq_len]
            attn_map = aggregate_attention_maps(
                decoder_outputs.attentions,
                aggregation_type=cfg.parameters.attention_aggregation,
                fusion_type=cfg.parameters.attention_head_fusion_type,
            )

            # metrics
            patched_img = model.patchify(pixel_values)
            loss = mae_loss(pred=logits, target=patched_img, use_norm_pix_loss=False).detach().cpu()

            entropy = shannon_entropy(attn_map).cpu()
            gini = gini_index(attn_map).cpu()

            max_attention = maximum_attention_weight(attn_map).cpu()
            kl_div = KL_divergence(attn_map).cpu()
            focus_score = attention_focus_score(attn_map).cpu()

            nl_entropy = -torch.log(entropy)
            nl_gini = -torch.log(gini)
            nl_kl_div= -torch.log(kl_div)

            loss_img = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)
            entropy_img = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1)
            gini_img = (gini * mask).sum(dim=-1) / mask.sum(dim=-1)
            # attn_spread_img = (attn_spread * mask).sum(dim=-1) / mask.sum(dim=-1)
            max_attention_img = (max_attention * mask).sum(dim=-1) / mask.sum(dim=-1)
            kl_div_img = (kl_div * mask).sum(dim=-1) / mask.sum(dim=-1)
            focus_score_img = (focus_score * mask).sum(dim=-1) / mask.sum(dim=-1)

            nl_entropy_img = -torch.log(entropy_img)
            nl_gini_img = -torch.log(gini_img)
            nl_kl_div_img = -torch.log(kl_div_img)

            measures_per_patch = {
                "loss": loss[mask_bool],
                "shannon_entropy": entropy[mask_bool],
                "gini_index": gini[mask_bool],
                "maximum_attention_weight": max_attention[mask_bool],
                "KL_divergence": kl_div[mask_bool],
                "attention_focus_score":focus_score[mask_bool],
                "neg_log_shannon_entropy": nl_entropy[mask_bool],
                "neg_log_gini_index": nl_gini[mask_bool],
                "neg_log_KL_divergence": nl_kl_div[mask_bool],
                # "attention_spread": attn_spread[mask],
            }
            table_per_patch = create_wandb_table(measures_per_patch)



            measures_per_img = {
                "loss": loss_img,
                "shannon_entropy": entropy_img,
                "gini_index": gini_img,
                "maximum_attention_weight": max_attention_img,
                "KL_divergence": kl_div_img,
                "attention_focus_score":focus_score_img,
                "neg_log_shannon_entropy": nl_entropy_img,
                "neg_log_gini_index": nl_gini_img,
                "neg_log_KL_divergence": nl_kl_div_img
                # "attention_spread": attn_spread_img,
            }
            table_per_img = create_wandb_table(measures_per_img)

            # log metrics per patch
            log_loss_vs_uncertainty_measure(table_per_patch, "loss", "shannon_entropy", title="Loss vs Shannon Entropy per patch")
            log_loss_vs_uncertainty_measure(table_per_patch, "loss", "gini_index", title="Loss vs Gini Index per patch")
            log_loss_vs_uncertainty_measure(table_per_patch, "loss", "neg_log_shannon_entropy", title="Loss vs -log(Shannon Entropy) per patch")
            log_loss_vs_uncertainty_measure(table_per_patch, "loss", "neg_log_gini_index", title="Loss vs -log(Gini Index) per patch")
            # log_loss_vs_uncertainty_measure(table_per_patch, "loss", "attention_spread", title="Loss vs Attention Spread per patch")
            log_loss_vs_uncertainty_measure(table_per_patch, "loss", "maximum_attention_weight", title="Loss vs maximum_attention_weight per patch")
            log_loss_vs_uncertainty_measure(table_per_patch, "loss", "KL_divergence", title="Loss vs KL_divergence per patch")
            log_loss_vs_uncertainty_measure(table_per_patch, "loss", "neg_log_KL_divergence", title="Loss vs -log(KL_divergence per patch)")
            log_loss_vs_uncertainty_measure(table_per_patch, "loss", "attention_focus_score", title="Loss vs attention_focus_score per patch")


            # log metrics per img
            log_loss_vs_uncertainty_measure(table_per_img, "loss", "shannon_entropy", title="Loss vs Shannon Entropy per img")
            log_loss_vs_uncertainty_measure(table_per_img, "loss", "gini_index", title="Loss vs Gini Index per img")
            log_loss_vs_uncertainty_measure(table_per_img, "loss", "neg_log_shannon_entropy", title="Loss vs -log(Shannon Entropy) per img")
            log_loss_vs_uncertainty_measure(table_per_img, "loss", "neg_log_gini_index", title="Loss vs -log(Gini Index) per img")
            # log_loss_vs_uncertainty_measure(table_per_img, "loss", "attention_spread", title="Loss vs Attention Spread per img")
            log_loss_vs_uncertainty_measure(table_per_img, "loss", "maximum_attention_weight", title="Loss vs maximum_attention_weight per img")
            log_loss_vs_uncertainty_measure(table_per_img, "loss", "KL_divergence", title="Loss vs KL_divergence per img")
            log_loss_vs_uncertainty_measure(table_per_img, "loss", "neg_log_KL_divergence", title="Loss vs -log(KL_divergence per patch)")
            log_loss_vs_uncertainty_measure(table_per_img, "loss", "attention_focus_score", title="Loss vs attention_focus_score per img")


if __name__ == "__main__":
    main()