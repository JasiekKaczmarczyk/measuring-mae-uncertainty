import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoImageProcessor, ViTMAEForPreTraining
from PIL import Image
import requests

from metrics import mae_loss, shannon_entropy, attention_spread

# policzyć entropię Shannona: output shape [batch_size, seq_len]
# policzyć attention spread: output shape [batch_size, seq_len]
# policzyć loss: output_shape [batch_size, seq_len]
# dataloader, huggingface datasets
# wizualizacje
# wandb
# r^2, korelacja między loss a naszą metryką

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

def main():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

    inputs = image_processor(images=image, return_tensors="pt")
    # outputs = model(**inputs, output_attentions=True)
    outputs = model.vit(**inputs)

    latent = outputs.last_hidden_state
    ids_restore = outputs.ids_restore
    mask = outputs.mask

    decoder_outputs = model.decoder(latent, ids_restore, output_attentions=True)
    logits = decoder_outputs.logits
    last_attn_map = decoder_outputs.attentions[-1].detach().cpu()
    last_attn_map = remove_cls_token(last_attn_map)
    patched_img = model.patchify(inputs["pixel_values"])

    loss = mae_loss(
        pred=logits,
        target=patched_img,
        mask=mask,
        use_norm_pix_loss=True,
    ).detach().cpu()

    entropy = shannon_entropy(last_attn_map)
    attn_spread = attention_spread(last_attn_map)

    # print(loss)
    # print(attn_spread)

    # plot_loss_vs_uncertainty_measure(loss, entropy)
    plot_loss_vs_uncertainty_measure(loss, attn_spread)
    # plot_attention_maps(last_attn_map[0])

if __name__ == "__main__":
    main()