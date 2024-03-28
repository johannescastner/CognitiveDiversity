import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from math import log, sqrt

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=1)

def get_embeddings(model, tokenizer, texts):
    max_length = 4096
    batch_dict = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return normalize_embeddings(embeddings)

def Entropy(pr):
    tot = 1.0 / sum(pr)
    return -sum([p * tot * log(p * tot, 2) for p in pr if p > 0])

def N_point_JSD(ls):
    n = len(ls)
    mix = sum(ls) / n
    return Entropy(mix) - (1.0 / n) * sum([Entropy(p) for p in ls])

def Diversity(ls):
    try:
        return sqrt(N_point_JSD(ls) / log(len(ls), 2))
    except:
        return np.nan

def main():
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')

    # Sample statements
    statements = [
        "I believe that climate change is a severe global issue.",
        "Climate change is not as serious as it is made out to be.",
        "Climate change is primarily driven by natural factors, not human activities.",
        "The consequences of climate change are exaggerated.",
        "Climate change is an urgent problem that requires immediate action."
    ]

    # Generate embeddings
    embeddings = get_embeddings(model, tokenizer, statements)
    embeddings_np = embeddings.detach().numpy()  # Convert to numpy array if needed

    # Normalize and calculate diversity
    embeddings_normalized = [e / np.linalg.norm(e, 2) for e in embeddings_np]
    diversity = Diversity(embeddings_normalized)
    print("Diversity of opinions:", diversity)

if __name__ == "__main__":
    main()
