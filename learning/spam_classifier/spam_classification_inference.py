from gpt_model import GPTModel
import torch
import tiktoken
from spam_dataset_loader import train_dataset


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )    
    # Alternatively, a more robust version is the following one, which handles the max_length=None case better
    # max_len = min(max_length,supported_context_length) if max_length else supported_context_length
    # input_ids = input_ids[:max_len]
    
    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

def load_fine_tuned_spam_classification_model(model_name, device):

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    #settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")


    model = GPTModel(BASE_CONFIG)
    for param in model.parameters():
        param.requires_grad = False


    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    #print("spam classification model:", model)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    model_state_dict = torch.load("fine_tuned_spam_classification_model/review_classifier.pth", map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict)
    return model


CHOOSE_MODEL = "gpt2-small (124M)"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_fine_tuned_spam_classification_model(CHOOSE_MODEL, device)

tokenizer = tiktoken.get_encoding("gpt2")

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(text_1, "->", classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(text_2, "->", classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

text_3 = (
    "I am not spam"
)

print(text_3, "->", classify_review(
    text_3, model, tokenizer, device, max_length=train_dataset.max_length
))

text_4 = (
    "you won a lottery of 1M USD"
)

print(text_4, "->", classify_review(
    text_4, model, tokenizer, device, max_length=train_dataset.max_length
))
