import torch
from model import GPTLanguageModel
import torch.optim as optim
from tqdm import tqdm
import tiktoken
import gc
import json

batch_size = 32   
block_size = 128   
max_iters = 3000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
patience = None
gradient_accumulation_steps = 2  
log = True
# torch.manual_seed(1337)

torch.cuda.empty_cache()
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_per_process_memory_fraction(0.8)  # Usa solo l'80% della GPU

def free_memory():
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

# Weights & Biases
if log:
    import wandb
    wandb.init(project="dante_gpt", config={
        "batch_size": batch_size,
        "block_size": block_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "learning_rate": learning_rate,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "gradient_accumulation_steps": gradient_accumulation_steps
    })
    config = wandb.config

def save_tokenizer_info(tokenizer, filepath):
    tokenizer_info = {
        'encoding_name': tokenizer.name,
        'vocab_size': tokenizer.n_vocab,
        'special_tokens': {
            'bos_token': getattr(tokenizer, 'bos_token', None),
            'eos_token': getattr(tokenizer, 'eos_token', None),
            'pad_token': getattr(tokenizer, 'pad_token', None),
        }
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_info, f, indent=2)

def load_and_tokenize_data():
    with open('data/inferno.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    save_tokenizer_info(enc, 'checkpoints/tokenizer_config.json')
    
    tokens = enc.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data))
    
    return (
        data[:n].to(device),  # train_data
        data[n:].to(device),  # val_data
        enc.n_vocab,          # vocab_size
        enc                   # tokenizer
    )

def save_checkpoint(model, optimizer, iter_num, val_loss, tokenizer, is_best=False):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter_num,
        'val_loss': val_loss,
        'model_config': {
            'vocab_size': model.token_embedding_table.weight.shape[0],
            'block_size': model.position_embedding_table.weight.shape[0],
            'n_embd': model.token_embedding_table.weight.shape[1],
            'n_layer': len(model.blocks),
            'n_head': model.blocks[0].sa.heads[0].key.weight.shape[0] // model.blocks[0].sa.heads[0].key.weight.shape[1]
        }
    }
    
    filepath = 'checkpoints/best_gpt_model.pth' if is_best else 'checkpoints/final_gpt_model.pth'
    torch.save(checkpoint, filepath)
    
    save_tokenizer_info(tokenizer, 'checkpoints/tokenizer_config.json')
   
train_data, val_data, vocab_size, tokenizer = load_and_tokenize_data()

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters=200):
    losses = {'train': 0.0, 'val': 0.0}
    model.eval()
    for split in ['train', 'val']:
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, batch_loss = model(X, Y)
            losses[split] += batch_loss.item()
        losses[split] /= eval_iters
    model.train()
    return losses

model = GPTLanguageModel(vocab_size, block_size, n_embd, n_layer, n_head, dropout).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler('cuda')  # mixed precision training

# Training loop con ottimizzazioni
best_val_loss = float('inf')
epochs_without_improvement = 0

for iter in tqdm(range(max_iters), desc="Training progress"):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            epochs_without_improvement = 0
            print("=> Saving the best model.")
            save_checkpoint(model, optimizer, iter, losses['val'], tokenizer, is_best=True)
        else:
            save_checkpoint(model, optimizer, iter, losses['val'], tokenizer, is_best=False)
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break
    
    # Gradient accumulation loop
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0
    
    for _ in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    if log:
        wandb.log({
            "Training Loss (Step)": accumulated_loss,
            "Step": iter,
        })

save_checkpoint(model, optimizer, max_iters, best_val_loss, tokenizer, is_best=False)

if log:
    wandb.finish()
