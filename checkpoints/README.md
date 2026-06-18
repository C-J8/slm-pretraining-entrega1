# Checkpoints

Arquivos `.pt`, `.pth` e `.safetensors` são ignorados pelo Git para evitar commits pesados.

Para a Entrega 1, hospede pelo menos um checkpoint parcial ou final no HuggingFace Hub ou Google Drive e coloque o link no README principal.

Para a Entrega Final, os checkpoints esperados são:

- `pretrain_2b/ckpt_best.pt` para pré-treino;
- `midtrain_best.pt` para mid-training;
- `sft_best.pt` para SFT final.

Checkpoints intermediários `*_step_*.pt` servem apenas para recuperação durante treinos longos. Depois de validar `*_best.pt` e `*_last.pt`, eles podem ser apagados localmente para liberar espaço.
