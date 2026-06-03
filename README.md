# Small Language Model - Entrega 1

## 0. Decisões oficiais do projeto

| Elemento | Escolha |
| --- | --- |
| Arquitetura | LLaMA-inspired decoder-only Transformer |
| Tamanho | Aproximadamente 100M parâmetros |
| Corpus | FineWeb-Edu sample-10BT |
| Idioma | Inglês |
| Tokenizer | GPT-2 BPE |
| Objetivo | Next-token prediction |
| Framework | PyTorch |
| Referência estrutural | nanoGPT, com código próprio/adaptado e creditado |

## 1. Objetivo

Este projeto implementa o pré-treino de um Small Language Model autorregressivo para previsão do próximo token. A Entrega 1 inclui preparação do corpus, implementação do modelo, treino, checkpoints, curva de loss e exemplos de geração.

## 2. Arquitetura

O modelo é um Transformer decoder-only inspirado na família LLaMA:

- atenção causal multi-head;
- RoPE como codificação posicional;
- RMSNorm antes da atenção e do bloco feed-forward;
- SwiGLU como ativação do feed-forward;
- pesos compartilhados entre embedding de tokens e camada final;
- objetivo de next-token prediction.

A configuração oficial está em [configs/pretrain_llama_100m.yaml](configs/pretrain_llama_100m.yaml). A configuração local usada na RTX 3060 está em [configs/pretrain_llama_100m_rtx3060_60k.yaml](configs/pretrain_llama_100m_rtx3060_60k.yaml).

Parâmetros do modelo local:

| Item | Valor |
| --- | ---: |
| Parâmetros | 109.392.384 |
| Layers | 10 |
| Hidden size | 768 |
| Heads | 12 |
| Block size local | 512 |
| Vocab size | 50.257 |

## 3. Corpus e tokenizer

Corpus escolhido:

- dataset: `HuggingFaceFW/fineweb-edu`;
- config: `sample-10BT`;
- split: `train`;
- idioma predominante: inglês;
- documentos processados localmente: 500.000.

O script [scripts/prepare_data.py](scripts/prepare_data.py) tokeniza o texto com GPT-2 BPE via `tiktoken` e salva os tokens em arquivos binários:

- `data/fineweb_edu_train_500m.bin`: 512.824.558 tokens;
- `data/fineweb_edu_val_500m.bin`: 5.133.236 tokens.

Cada documento recebe o token `<|endoftext|>` ao final, usando o ID nativo do GPT-2.

## 4. Treino

Configuração final local:

| Item | Valor |
| --- | ---: |
| GPU | RTX 3060 12 GB |
| Batch size | 1 |
| Gradient accumulation | 16 |
| Tokens por step | 8.192 |
| Max steps | 60.000 |
| Tokens vistos | 491.520.000 |
| Learning rate | `1e-5` |
| Scheduler | constant |
| Optimizer | AdamW |
| Weight decay | 0.1 |
| Dtype | float16 |

Pelas Chinchilla Scaling Laws, um modelo de aproximadamente 109M parâmetros teria como referência ideal algo próximo de 2B tokens. Nesta entrega, por limitação de tempo e treino local em uma RTX 3060, foi usado um pré-treino parcial com 491,5M tokens vistos.

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

Exemplos de geração:

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

## 6. Geração

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

Checkpoints pesados não devem ser versionados diretamente no GitHub. Para a entrega, hospede `checkpoints/ckpt_best.pt` ou `checkpoints/ckpt_last.pt` no HuggingFace Hub ou Google Drive e preencha:

- Link do checkpoint: [HuggingFace Hub - ckpt_best.pt](https://huggingface.co/C1Junin2/slm-pretraining-entrega1)

## 8. Créditos

Este projeto usa PyTorch, HuggingFace Datasets, tiktoken, NumPy e Matplotlib. A organização geral do pipeline de pré-treino foi inspirada pelo nanoGPT, com implementação própria/adaptada para uma arquitetura LLaMA-inspired.
