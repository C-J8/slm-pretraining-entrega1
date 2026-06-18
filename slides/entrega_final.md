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
- Dados locais: 2.000.000 documentos.
- Tokens processados: 2.000.322.560.
- Hardware: RTX 3060 12 GB + 2x RTX A5000 24 GB.

## 4. Scaling Laws

- Chinchilla recomenda aproximadamente 20 tokens por parametro.
- Para 109M parametros, o ideal seria perto de 2B tokens.
- O treino final chegou a aproximadamente 2B tokens.
- A continuacao 1B -> 2B usou DDP com 2 GPUs no servidor PUCRS.

## 5. Resultados de Pre-treino

- Loss final de treino: 3.8945 no step 183.140.
- Loss final de validacao: 3.9927.
- Melhor loss de validacao: 3.8937 no step 155.500.
- PPL aproximada no melhor ponto: 49.09.

## 6. Mid-training

- Dataset: SmolTalk, subset `all`.
- Objetivo: especializar o modelo em dados conversacionais e de raciocinio.
- Inicializacao: pesos do melhor checkpoint de pre-treino 2B.
- Template: `System:`, `User:`, `Assistant:`.
- Melhor validacao: loss 2.1819 no step 8.200.

## 7. SFT

- Inicializacao: melhor checkpoint de mid-training.
- Mesmo template conversacional.
- Loss mask com `-100` nos tokens de prompt.
- Apenas tokens de resposta do assistente contribuem para a loss.
- Melhor validacao: loss 2.1770 no step 3.400.

## 8. Avaliacao

- Perplexidade no split de validacao.
- Benchmark de multipla escolha por log-likelihood.
- Benchmarks suportados no codigo: HellaSwag, ARC-Easy, ARC-Challenge, PIQA e WinoGrande.
- SFT PPL em FineWeb-Edu val: 101.77.
- HellaSwag com 200 exemplos: 34.5% de acuracia.

## 9. Demo Chat

- Interface em Streamlit.
- Carrega `checkpoints/sft_best.pt`.
- Permite ajustar temperatura, top-k e maximo de tokens.
- Mantem historico de conversa durante a sessao.

## 10. Discussao

- Funcionou bem: pipeline modular, retomada de treino, DDP e curva de loss decrescente.
- Limitacao principal: qualidade conversacional ainda instavel em um modelo de 109M parametros.
- Com mais recursos: ampliar benchmarks, comparar estagios e melhorar dataset/hiperparametros de SFT.
