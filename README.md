# GPT Language Model Training on Dante's Inferno

## Overview
This project trains a Generative Pre-trained Transformer (GPT) language model on Dante Alighieri’s *Inferno*, the first part of the *Divine Comedy*. The model is designed to generate text in the style of Dante, aiming to capture the poetic structure and unique vocabulary of the original work.

The training script (`train.py`) includes functionalities such as data preprocessing, model training, evaluation, and text generation. The model architecture is defined in `model.py`.


## Data Preparation
`data/inferno.txt` contains the full text of *Inferno*. The training script will tokenize this text and split it into training, validation, and test sets:

- **Training set**: 80%
- **Validation set**: 10%
- **Test set**: 10%

## Model Architecture
The model is based on the GPT architecture with the following features:
- **Embedding size** (`n_embd`): 384
- **Number of heads** (`n_head`): 6
- **Number of layers** (`n_layer`): 6
- **Dropout**: 20%

The model consists of:
1. **Multi-Head Self-Attention**: Captures long-range dependencies in the text.
2. **Feed-Forward Network**: Processes the output of the attention mechanism.
3. **Layer Normalization**: Stabilizes training.
4. **Positional Embeddings**: Encodes the position of tokens.

Refer to `model.py` for full implementation details.

## Training
To train the model, run:
```bash
python train.py
```

### Sample Output
The model can generate text in the style of Dante after being trained. For example:
```python
python generate.py \
--prompt "Allor" \
--checkpoints 'checkpoints/final_gpt_model.pth' \
--max-tokens 1000
```

```
Quando fuor giunti, assai con l'occhio bieco
  mi rimiraron sanza far parola;
  poi si volsero in se', e dicean seco:

<<Costui par vivo a l'atto de la gola;
  e s'e' son morti, per qual privilegio
  vanno scoperti de la grave stola?>>.

Poi disser me: <<O Tosco, ch'al collegio
  de l'ipocriti tristi se' venuto,
  dir chi tu se' non avere in dispregio>>.

E io a loro: <<I' fui nato e cresciuto
  sovra 'l bel fiume d'Arno a la gran villa,
  e son col corpo ch'i' ho sempre avuto.

Ma voi chi siete, a cui tanto distilla
  quant'i' veggio dolor giu` per le guance?
  e che pena e` in voi che si` sfavilla?>>.

E l'un rispuose a me: <<Le cappe rance
  son di piombo si` grosse, che li pesi
  fan cosi` cigolar le lor bilance.

Frati godenti fummo, e bolognesi;
  io Catalano e questi Loderingo
  nomati, e da tua terra insieme presi,

come suole esser tolto un uom solingo,
  per conservar sua pace; e fummo tali,
  ch'ancor si pare intorno dal Gardingo>>.
```

## Checkpoints
- **Best Model**: Saved at `checkpoints/best_gpt_model.pth`.
- **Final Model**: Saved at `checkpoints/final_gpt_model.pth`.

The `tokenizer_config.json` contains information about the tokenizer used for encoding the text.
