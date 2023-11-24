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

from dataset import MAEDataset
from metrics import mae_loss, shannon_entropy, gini_index, attention_spread

# policzyć entropię Shannona: output shape [batch_size, seq_len]
# policzyć attention spread: output shape [batch_size, seq_len]
# policzyć loss: output_shape [batch_size, seq_len]
# dataloader, huggingface datasets
# wizualizacje
# wandb
# r^2, korelacja między loss a naszą metryką

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
    return attention[:, :, 1:, 1:]

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
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader

def log_loss_vs_uncertainty_measure(loss: torch.Tensor, measure: torch.Tensor, title: str):
    loss = torch.flatten(loss)
    measure = torch.flatten(measure)

    data = [[x, y] for (x, y) in zip(loss, measure)]
    table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log({title : wandb.plot.scatter(table, "x", "y", title=title)})


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def main(cfg: OmegaConf):
    makedir_if_not_exists(cfg.parameters.log_dir)

    wandb.login()

    wandb.init(
        project="mae-uncertainty",
        name=cfg.parameters.run_name,
        dir=cfg.parameters.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(cfg.parameters.device)

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
            mask = outputs.mask.to(torch.long)

            decoder_outputs = model.decoder(latent, ids_restore, output_attentions=True)
            logits = decoder_outputs.logits
            # get last attention map
            last_attn_map = decoder_outputs.attentions[-1].detach()
            last_attn_map = remove_cls_token(last_attn_map)

            # metrics
            patched_img = model.patchify(pixel_values)
            loss = mae_loss(pred=logits, target=patched_img, use_norm_pix_loss=False)[mask].detach().cpu()

            entropy = shannon_entropy(last_attn_map)[mask].cpu()
            gini = gini_index(last_attn_map)[mask].cpu()
            attn_spread = attention_spread(last_attn_map)[mask].cpu()

            # log metrics per patch
            log_loss_vs_uncertainty_measure(loss, entropy, title="Loss vs Shannon Entropy per patch")
            log_loss_vs_uncertainty_measure(loss, gini, title="Loss vs Gini Index per patch")
            log_loss_vs_uncertainty_measure(loss, attn_spread, title="Loss vs Attention Spread per patch")


if __name__ == "__main__":
    main()