import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoImageProcessor, ViTMAEModel
from PIL import Image
import requests

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

def main():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)

    print(outputs.attentions[-1].shape)
    # shape ([batch_size, num_heads, seq_len, seq_len], ...)
    # attentions = outputs.attentions
    # attn = attentions[-1][0].detach().cpu()

    # plot_attention_maps(attn)

if __name__ == "__main__":
    main()