# Small Language Model - Entrega Final

## 0. Decisões oficiais do projeto

| Elemento | Escolha |
| --- | --- |
| Arquitetura | LLaMA-inspired decoder-only Transformer |
| Tamanho | Aproximadamente 100M parâmetros |
| Corpus de pré-treino | FineWeb-Edu sample-10BT |
| Mid-training/SFT | SmolTalk |
| Idioma | Inglês |
| Tokenizer | GPT-2 BPE |
| Objetivo | Next-token prediction |
| Framework | PyTorch |
| Referência estrutural | nanoGPT, com código próprio/adaptado e creditado |

## 1. Objetivo

Este projeto implementa o pipeline de um Small Language Model autorregressivo para previsão do próximo token. O repositório cobre pré-treino, mid-training, supervised fine-tuning (SFT), avaliação por perplexidade/benchmark de múltipla escolha e uma demo em modo chat.

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

Corpus de pré-treino:

- dataset: `HuggingFaceFW/fineweb-edu`;
- config: `sample-10BT`;
- split: `train`;
- idioma predominante: inglês;
- documentos processados localmente: 500.000.

O script [scripts/prepare_data.py](scripts/prepare_data.py) tokeniza o texto com GPT-2 BPE via `tiktoken` e salva os tokens em arquivos binários:

- `data/fineweb_edu_train_500m.bin`: 512.824.558 tokens;
- `data/fineweb_edu_val_500m.bin`: 5.133.236 tokens.

Cada documento recebe o token `<|endoftext|>` ao final, usando o ID nativo do GPT-2.

## 4. Pré-treino

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

### 4.1 Continuação opcional para 1B e 2B tokens

As configs abaixo mantêm `batch_size=1`, `block_size=512` e `gradient_accumulation_steps=16`, ou seja, continuam usando 8.192 tokens por step. Os logs novos incluem uma coluna `timestamp` para medir a duração real do treino.

Como já existe um corpus local com aproximadamente 500M tokens únicos, o fluxo correto para 2B é preparar apenas a fatia adicional de aproximadamente 1,5B tokens e depois concatenar os binários.

Preparar a fatia adicional pulando os primeiros 500.000 documentos já usados:

```bash
python scripts/prepare_data.py \
  --config configs/pretrain_fineweb_extra_1_5b.yaml \
  --skip-documents 500000 \
  --max-documents 1500000
```

Combinar os 500M existentes com a fatia adicional:

```bash
python scripts/merge_token_bins.py \
  --output data/fineweb_edu_train_2b.bin \
  --inputs data/fineweb_edu_train_500m.bin data/fineweb_edu_train_extra_1_5b.bin

python scripts/merge_token_bins.py \
  --output data/fineweb_edu_val_2b.bin \
  --inputs data/fineweb_edu_val_500m.bin data/fineweb_edu_val_extra_1_5b.bin
```

Após essa etapa, os arquivos esperados são:

- `data/fineweb_edu_train_2b.bin`
- `data/fineweb_edu_val_2b.bin`

Para continuar de 491,5M tokens até aproximadamente 1B tokens vistos:

```bash
python src/train.py \
  --config configs/pretrain_llama_100m_rtx3060_1b.yaml \
  --resume checkpoints/ckpt_last.pt
```

Saídas principais:

- `checkpoints/pretrain_1b/ckpt_best.pt`
- `checkpoints/pretrain_1b/ckpt_last.pt`
- `outputs/train_log_60k_1b.csv`

Para continuar de 1B até aproximadamente 2B tokens vistos:

```bash
python src/train.py \
  --config configs/pretrain_llama_100m_rtx3060_2b.yaml \
  --resume checkpoints/pretrain_1b/ckpt_last.pt
```

Saídas principais:

- `checkpoints/pretrain_2b/ckpt_best.pt`
- `checkpoints/pretrain_2b/ckpt_last.pt`
- `outputs/train_log_1b_2b.csv`

Observação: após um pré-treino melhor, o ideal metodológico é refazer mid-training e SFT a partir do novo melhor checkpoint.

### 4.2 Servidor PUCRS com 2 GPUs

No servidor com 2x RTX A5000 24GB, use DDP com `torchrun`. O código mantém o mesmo `batch_size=1` por GPU para reduzir risco de OOM, mas dobra os tokens globais por step.

Arquivos grandes que precisam estar no servidor:

- `data/fineweb_edu_train_2b.bin`
- `data/fineweb_edu_val_2b.bin`
- `checkpoints/pretrain_1b/ckpt_last.pt`

Comando recomendado:

```bash
torchrun --standalone --nproc_per_node=2 src/train.py \
  --config configs/pretrain_llama_100m_a5000_2gpu_2b.yaml \
  --resume checkpoints/pretrain_1b/ckpt_last.pt
```

Essa config usa:

| Item | Valor |
| --- | ---: |
| GPUs | 2 |
| Batch por GPU | 1 |
| Gradient accumulation | 16 |
| Tokens globais por step | 16.384 |
| Step inicial esperado | 122.101 |
| Step final | 183.150 |
| Tokens finais vistos | ~2B |

Para rodar sem perder o processo ao desconectar do SSH:

```bash
tmux new -s prof2
torchrun --standalone --nproc_per_node=2 src/train.py \
  --config configs/pretrain_llama_100m_a5000_2gpu_2b.yaml \
  --resume checkpoints/pretrain_1b/ckpt_last.pt
```

Monitoramento:

```bash
tail -f outputs/train_log_1b_2b.csv
watch -n 5 nvidia-smi
```

## 5. Mid-training

O mid-training continua a partir do checkpoint de pré-treino usando dados conversacionais e de raciocínio do SmolTalk (`HuggingFaceTB/smoltalk`, subset `all`). Todas as mensagens são convertidas para um template simples:

```text
System:
...
User:
...
Assistant:
...
```

Preparar os dados:

```bash
python scripts/prepare_chat_data.py --config configs/midtrain_smoltalk.yaml --phase mid --max-examples 50000
```

Treinar a partir do checkpoint de pré-treino:

```bash
python src/finetune.py \
  --config configs/midtrain_smoltalk.yaml \
  --init-checkpoint checkpoints/ckpt_best.pt
```

Checkpoints esperados:

- `checkpoints/midtrain_best.pt`
- `checkpoints/midtrain_last.pt`

Resultados obtidos:

| Checkpoint | Step | Train loss | Val loss |
| --- | ---: | ---: | ---: |
| `checkpoints/midtrain_best.pt` | 6.200 | 2.4482 | 2.2997 |
| `checkpoints/midtrain_last.pt` | 10.000 | 2.5875 | 2.4792 |

Curva de loss:

- [outputs/midtrain_loss_curve.png](outputs/midtrain_loss_curve.png)

## 6. Supervised fine-tuning (SFT)

O SFT usa o mesmo formato conversacional, mas com loss mask: tokens de `System:` e `User:` recebem label `-100`; apenas os tokens das respostas `Assistant:` contribuem para a cross-entropy.

Preparar os dados:

```bash
python scripts/prepare_chat_data.py --config configs/sft_smoltalk.yaml --phase sft --max-examples 50000
```

Treinar a partir do melhor checkpoint de mid-training:

```bash
python src/finetune.py \
  --config configs/sft_smoltalk.yaml \
  --init-checkpoint checkpoints/midtrain_best.pt
```

Checkpoint esperado para a entrega final:

- `checkpoints/sft_best.pt`
- `checkpoints/sft_last.pt`

Resultados obtidos:

| Checkpoint | Step | Train loss | Val loss |
| --- | ---: | ---: | ---: |
| `checkpoints/sft_best.pt` | 3.400 | 2.4306 | 2.3261 |
| `checkpoints/sft_last.pt` | 5.000 | 2.3872 | 2.6437 |

Curva de loss:

- [outputs/sft_loss_curve.png](outputs/sft_loss_curve.png)

## 7. Avaliação

### 7.1 Perplexidade

Perplexidade é calculada como `exp(loss)` sobre um split de validação.

```bash
python src/evaluate.py perplexity \
  --checkpoint checkpoints/sft_best.pt \
  --data data/fineweb_edu_val_500m.bin \
  --eval-iters 100 \
  --batch-size 1 \
  --output outputs/sft_perplexity.json
```

Resultado obtido com `checkpoints/sft_best.pt` sobre `data/fineweb_edu_val_500m.bin`:

| Loss | PPL |
| ---: | ---: |
| 4.8556 | 128.46 |

Para exportar a curva de validação do pré-treino:

```bash
python src/evaluate.py val_curve \
  --log outputs/train_log_0_60k.csv \
  --output outputs/pretrain_val_curve.json
```

### 7.2 Benchmark de múltipla escolha

A avaliação escolhe a alternativa com maior log-verossimilhança condicional dado o contexto. Benchmarks suportados:

- `hellaswag`
- `arc_easy`
- `arc_challenge`
- `piqa`
- `winogrande`

Exemplo com HellaSwag:

```bash
python src/evaluate.py multiple_choice \
  --checkpoint checkpoints/sft_best.pt \
  --benchmark hellaswag \
  --split validation \
  --max-examples 200 \
  --output outputs/sft_hellaswag_200.json
```

Resultado obtido em 200 exemplos do HellaSwag:

| Benchmark | Exemplos | Acertos | Acurácia |
| --- | ---: | ---: | ---: |
| HellaSwag | 200 | 67 | 33,5% |

## 8. Demo chat

A demo em modo chat usa Streamlit, carrega o checkpoint final de SFT e permite ajustar temperatura, top-k e número máximo de tokens.

```bash
streamlit run app.py
```

Por padrão, a interface procura:

```text
checkpoints/sft_best.pt
```

O caminho pode ser alterado na barra lateral.

## 9. Slides

Os slides da apresentação estão em:

- [slides/entrega_final.md](slides/entrega_final.md)
- [slides/entrega_final.pdf](slides/entrega_final.pdf)

## 10. Resultados de pré-treino

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

Perplexidade aproximada no último ponto de validação do pré-treino:

| Step | Val loss | PPL |
| ---: | ---: | ---: |
| 60.000 | 4.2957 | 73.38 |

O arquivo [outputs/pretrain_val_curve.json](outputs/pretrain_val_curve.json) exporta todos os pontos de validação com perplexidade.

## 11. Geração

```bash
python src/generate.py \
  --checkpoint checkpoints/ckpt_best.pt \
  --prompt "The future of artificial intelligence is" \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 50 \
  --output outputs/samples_best_60k.txt
```

## 12. Checkpoints

Checkpoints pesados não devem ser versionados diretamente no GitHub. Para a entrega, hospede `checkpoints/ckpt_best.pt` ou `checkpoints/ckpt_last.pt` no HuggingFace Hub ou Google Drive.

- Link do checkpoint de pré-treino: [HuggingFace Hub - ckpt_best.pt](https://huggingface.co/C1Junin2/slm-pretraining-entrega1)
- Link do checkpoint final SFT: preencher após o treino final.

Para economizar espaço local, checkpoints intermediários como `midtrain_step_*.pt` e `sft_step_*.pt` podem ser removidos depois de confirmar que `*_best.pt` e `*_last.pt` existem.

## 13. Créditos

Este projeto usa PyTorch, HuggingFace Datasets, tiktoken, NumPy, Matplotlib e Streamlit. A organização geral do pipeline de pré-treino foi inspirada pelo nanoGPT, com implementação própria/adaptada para uma arquitetura LLaMA-inspired.
