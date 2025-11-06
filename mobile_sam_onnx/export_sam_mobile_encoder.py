import torch
from mobile_sam import sam_model_registry

checkpoint = "/home/logan78/Desktop/adobe/models/mobile_sam.pt"

# Load MobileSAM
model = sam_model_registry["vit_t"](checkpoint=checkpoint)
model.eval()

encoder = model.image_encoder
decoder = model.mask_decoder
image_pe = model.prompt_encoder.get_dense_pe()  # ✅ positional embedding

# ------------------ Export Encoder ------------------
dummy_image = torch.randn(1, 3, 1024, 1024)

torch.onnx.export(
    encoder,
    dummy_image,
    "mobile_sam_encoder.onnx",
    input_names=["image"],
    output_names=["image_embedding"],
    opset_version=17,
    do_constant_folding=True,
)
print("✅ Encoder saved as mobile_sam_encoder.onnx")

# ------------------ Wrap Decoder ------------------
class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder, image_pe):
        super().__init__()
        self.decoder = decoder
        self.image_pe = image_pe

    def forward(self, image_embedding, sparse_prompt_embeddings, dense_prompt_embeddings):
        # multimask_output=False (single best mask)
        masks, iou_scores = self.decoder(
            image_embeddings=image_embedding,
            image_pe=self.image_pe,                      # ✅ REQUIRED
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )
        return masks, iou_scores

wrapper = DecoderWrapper(decoder, image_pe)

# Dummy inputs for tracing
image_embedding = encoder(dummy_image)
sparse = torch.zeros(1, 2, decoder.transformer_dim)
dense = torch.zeros(1, decoder.transformer_dim, 64, 64)

# ------------------ Export Decoder ------------------
torch.onnx.export(
    wrapper,
    (image_embedding, sparse, dense),
    "mobile_sam_decoder.onnx",
    input_names=["image_embedding", "sparse_prompt", "dense_prompt"],
    output_names=["masks", "iou_scores"],
    opset_version=17,
    do_constant_folding=True,
)

print("🎉 Decoder saved as mobile_sam_decoder.onnx")
