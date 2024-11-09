import torch
from model import GPTLanguageModel
import tiktoken
import json
from argparse import ArgumentParser

def load_tokenizer_config():
    with open('checkpoints/tokenizer_config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_and_tokenizer(checkpoint_path):
    tokenizer_config = load_tokenizer_config()
    tokenizer = tiktoken.get_encoding(tokenizer_config['encoding_name'])
    
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    model_config = checkpoint['model_config']
    model = GPTLanguageModel(
        vocab_size=model_config['vocab_size'],
        block_size=model_config['block_size'],
        n_embd=model_config['n_embd'],
        n_layer=model_config['n_layer'],
        n_head=max(model_config['n_head'], 6),
        dropout=0.0  
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt="", max_tokens=1000, temperature=1.0):
    """Genera testo usando il modello e il tokenizer"""
    model.eval()
    
    if prompt:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    generated_tokens = model.generate(context, max_tokens, tokenizer)[0].tolist()
    
    return tokenizer.decode(generated_tokens)

if __name__ == '__main__':
    parser = ArgumentParser(description='Text generation using a pre-trained GPT model.')

    parser.add_argument('--prompt', type=str, default="Nel mezzo del cammin di nostra vita",
                        help='The prompt to start the generation from.')
    parser.add_argument('--checkpoints', type=str, default='checkpoints/best_gpt_model.pth',
                        help='Path to the checkpoint file containing the model weights.')
    parser.add_argument('--max-tokens', type=int, default=100)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model_and_tokenizer(args.checkpoints)
    model.to(device)
    
    print(args.prompt)
    generated_text = generate_text(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)

