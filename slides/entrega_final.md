# Small Language Model - Entrega Final

## 1. Objetivo

- Construir um Small Language Model autorregressivo completo.
- Cobrir pre-treino, mid-training, SFT, avaliacao e demo chat.
- Reproduzir em escala reduzida o pipeline de desenvolvimento de LLMs.

## 2. Arquitetura

- Decoder-only Transformer inspirado na familia LLaMA.
- 109.392.384 parametros treinaveis.
- 10 camadas, hidden size 768, 12 heads, block size 512.
- RoPE, RMSNorm, SwiGLU e pesos compartilhados entre embedding e cabeca final.

## 3. Pre-treino

- Dataset: FineWeb-Edu `sample-10BT`.
- Tokenizer: GPT-2 BPE.
- Dados locais: 500.000 documentos.
- Tokens processados: 491.520.000.
- Hardware: RTX 3060 12 GB.

## 4. Scaling Laws

- Chinchilla recomenda aproximadamente 20 tokens por parametro.
- Para 109M parametros, o ideal seria perto de 2B tokens.
- O treino local usou 491,5M tokens por limite de tempo e hardware.
- A decisao priorizou pipeline completo e checkpoint treinado.

## 5. Resultados de Pre-treino

- Loss final de treino: 4.1946.
- Loss final de validacao: 4.2957.
- Melhor loss de validacao: 4.2281 no step 58.200.
- PPL aproximada no ultimo ponto: 73.38.

## 6. Mid-training

- Dataset: SmolTalk, subset `all`.
- Objetivo: especializar o modelo em dados conversacionais e de raciocinio.
- Inicializacao: pesos do melhor checkpoint de pre-treino.
- Template: `System:`, `User:`, `Assistant:`.
- Melhor validacao: loss 2.2997 no step 6.200.

## 7. SFT

- Inicializacao: melhor checkpoint de mid-training.
- Mesmo template conversacional.
- Loss mask com `-100` nos tokens de prompt.
- Apenas tokens de resposta do assistente contribuem para a loss.
- Melhor validacao: loss 2.3261 no step 3.400.

## 8. Avaliacao

- Perplexidade no split de validacao.
- Benchmark de multipla escolha por log-likelihood.
- Benchmarks suportados no codigo: HellaSwag, ARC-Easy, ARC-Challenge, PIQA e WinoGrande.
- SFT PPL em FineWeb-Edu val: 128.46.
- HellaSwag com 200 exemplos: 33.5% de acuracia.

## 9. Demo Chat

- Interface em Streamlit.
- Carrega `checkpoints/sft_best.pt`.
- Permite ajustar temperatura, top-k e maximo de tokens.
- Mantem historico de conversa durante a sessao.

## 10. Discussao

- Funcionou bem: pipeline modular, checkpoint de pre-treino, curva de loss decrescente.
- Limitacao principal: volume de tokens abaixo do ideal Chinchilla.
- Com mais recursos: treinar ate 2B tokens, ampliar benchmarks e comparar estagios.
