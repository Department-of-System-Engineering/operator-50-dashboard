import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # fontos: a legeslegtetején legyen

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Ha NINCS szükség GPU-ra, a legegyszerűbb:
DEVICE = "cpu"

# Ha szeretnél automatikus detektálást, csak akkor:
# import torch
# if torch.cuda.is_available():
#     DEVICE = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     DEVICE = "mps"
# else:
#     DEVICE = "cpu"

# Ésszerű batch méret (16–64). A 768 NEM batch!
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=DEVICE,            # <- fontos
    embed_batch_size=32       # <- ésszerű batch
)

def generate_embeddings(text: str, metadata: dict = {}):
    return embed_model.get_text_embedding(text)