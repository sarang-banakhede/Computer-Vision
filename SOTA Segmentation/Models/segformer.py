from transformers import SegformerConfig, SegformerForSemanticSegmentation
from typing import Tuple

def get_segformer_model(num_labels: int, image_size: Tuple[int, int]):
    config = SegformerConfig(
    num_labels=num_labels,
    image_size=image_size,
    num_channels=3,
    num_encoder_blocks=4,
    depths=[3, 4, 6, 3],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[64, 128, 320, 512],
    num_attention_heads=[1, 2, 5, 8],
    patch_sizes=[7, 3, 3, 3],
    stride_sizes=[4, 2, 2, 2],
    decoder_hidden_size=768,  
    mlp_ratio=4,
    dropout=0.0,
    attention_dropout=0.0,
    classifier_dropout=0.1
)

    model = SegformerForSemanticSegmentation(config)
    return model

if __name__ == "__main__":
    pass