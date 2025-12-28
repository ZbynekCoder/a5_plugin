## Files
- `a5_core.py`: A5 group construction, dataset, and evaluation helper.
- `models.py`: `GPT2FrozenStatePlugin` (frozen GPT-2 + trainable state channel).
- `train.py`: Training script with key ablations:
  - `--train_inject_mode clean|final|prev|none`
  - `--state_stride`, `--stride_mode`, `--random_phase_shift`
  - `--mid_once`, `--mid_pos`

## Example
```bash
python train.py \
  --device cuda:6 \
  --train_inject_mode clean \
  --inject_style input_add \
  --mid_once --mid_pos -1 --mid_pos_mode batch \
  --d_state 128 --inject_layer 8 \
  --lr 5e-3 --schedule 64 --steps_per_stage 1000 \
  --eval_every 200 --eval_inject_modes clean,final
