# Small Language Model - Entrega 1

## 0. Decisoes oficiais do projeto

| Elemento | Escolha |
| --- | --- |
| Arquitetura | LLaMA-inspired decoder-only Transformer |
| Tamanho | Aproximadamente 100M parametros |
| Corpus | FineWeb-Edu sample-10BT |
| Idioma | Ingles |
| Tokenizer | GPT-2 BPE |
| Objetivo | Next-token prediction |
| Framework | PyTorch |
| Referencia estrutural | nanoGPT, com codigo proprio/adaptado e creditado |

## 1. Objetivo

Este projeto implementa o pre-treino de um Small Language Model autorregressivo para previsao do proximo token. A Entrega 1 inclui preparacao do corpus, implementacao do modelo, treino, checkpoints, curva de loss e exemplos de geracao.

## 2. Arquitetura

O modelo e um Transformer decoder-only inspirado na familia LLaMA:

- atencao causal multi-head;
- RoPE como codificacao posicional;
- RMSNorm antes da atencao e do bloco feed-forward;
- SwiGLU como ativacao do feed-forward;
- pesos compartilhados entre embedding de tokens e camada final;
- objetivo de next-token prediction.

A configuracao oficial esta em [configs/pretrain_llama_100m.yaml](configs/pretrain_llama_100m.yaml). A configuracao local usada na RTX 3060 esta em [configs/pretrain_llama_100m_rtx3060_60k.yaml](configs/pretrain_llama_100m_rtx3060_60k.yaml).

Parametros do modelo local:

| Item | Valor |
| --- | ---: |
| parametros | 109.392.384 |
| layers | 10 |
| hidden size | 768 |
| heads | 12 |
| block size local | 512 |
| vocab size | 50.257 |

## 3. Corpus e tokenizer

Corpus escolhido:

- dataset: `HuggingFaceFW/fineweb-edu`
- config: `sample-10BT`
- split: `train`
- idioma predominante: ingles
- documentos processados localmente: 500.000

O script [scripts/prepare_data.py](scripts/prepare_data.py) tokeniza o texto com GPT-2 BPE via `tiktoken` e salva os tokens em arquivos binarios:

- `data/fineweb_edu_train_500m.bin`: 512.824.558 tokens;
- `data/fineweb_edu_val_500m.bin`: 5.133.236 tokens.

Cada documento recebe o token `<|endoftext|>` ao final, usando o id nativo do GPT-2.

## 4. Treino

Configuracao final local:

| Item | Valor |
| --- | ---: |
| GPU | RTX 3060 12 GB |
| batch size | 1 |
| gradient accumulation | 16 |
| tokens por step | 8.192 |
| max steps | 60.000 |
| tokens vistos | 491.520.000 |
| learning rate | `1e-5` |
| scheduler | constant |
| optimizer | AdamW |
| weight decay | 0.1 |
| dtype | float16 |

Como executar:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --config configs/pretrain_llama_100m.yaml --max-documents 500000
python scripts/run_smoke_test.py
python src/train.py --config configs/pretrain_llama_100m_rtx3060_60k.yaml
```

Para continuar a partir de um checkpoint existente:

```bash
python src/train.py --config configs/pretrain_llama_100m_rtx3060_60k.yaml --resume checkpoints/ckpt_last.pt
```

## 5. Resultados

Log final:

- [outputs/train_log_0_60k.csv](outputs/train_log_0_60k.csv)

Curva de loss:

- [outputs/loss_curve_0_60k.png](outputs/loss_curve_0_60k.png)

Exemplos de geracao:

- [outputs/samples_best_60k.txt](outputs/samples_best_60k.txt)

Resumo da loss:

| Step | Tokens vistos | Train loss | Val loss |
| ---: | ---: | ---: | ---: |
| 10.000 | 81.920.000 | 4.6403 | 4.6868 |
| 15.000 | 122.880.000 | 4.6170 | 4.6905 |
| 35.000 | 286.720.000 | 4.4181 | 4.5173 |
| 60.000 | 491.520.000 | 4.1946 | 4.2957 |

Melhor checkpoint observado:

| Checkpoint | Step | Train loss | Val loss |
| --- | ---: | ---: | ---: |
| `checkpoints/ckpt_best.pt` | 58.200 | 4.2186 | 4.2281 |

Checkpoint final:

| Checkpoint | Step | Train loss | Val loss |
| --- | ---: | ---: | ---: |
| `checkpoints/ckpt_last.pt` | 60.000 | 4.1946 | 4.2957 |

## 6. Geracao

```bash
python src/generate.py \
  --checkpoint checkpoints/ckpt_best.pt \
  --prompt "The future of artificial intelligence is" \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 50 \
  --output outputs/samples_best_60k.txt
```

## 7. Checkpoint

Checkpoints pesados nao devem ser versionados diretamente no GitHub. Para a entrega, hospedar `checkpoints/ckpt_best.pt` ou `checkpoints/ckpt_last.pt` no HuggingFace Hub ou Google Drive e preencher:

- Link do checkpoint: `PREENCHER_APOS_UPLOAD`

## 8. Creditos

Este projeto usa PyTorch, HuggingFace Datasets, tiktoken, NumPy e Matplotlib. A organizacao geral do pipeline de pre-treino foi inspirada pelo nanoGPT, com implementacao propria/adaptada para uma arquitetura LLaMA-inspired.
