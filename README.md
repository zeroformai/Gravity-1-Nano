# Gravity-1

Gravity-1 is a local Q/A chat model aimed at daily-use assistance. New training
defaults to a denser LLaMA/Qwen-style decoder stack: RMSNorm, RoPE, SwiGLU,
full multi-head self-attention, KV cache generation, tied embeddings, mixed
precision training, and answer-only supervised loss. Smaller GQA presets and
legacy LayerNorm + fused-MHA checkpoints are still loadable.

## Corpus

Build and validate the main corpus:

```bash
python build_corpus.py
python validate_corpus.py
```

## Preference Tuning (DeepSeek + Minimal DPO)

Gravity can collect preference data, let DeepSeek judge candidate answers, and
then tune from those pairs with a very small DPO run. This is the recommended
first step before heavier PPO-style reinforcement learning.

Generate a tiny candidate batch:

```bash
python generate_candidates.py --checkpoint checkpoints_gravity120/model.pt --tokenizer checkpoints_gravity120/tokenizer.json --max_prompts 500 --num_candidates 1 --max_new_tokens 40 --resume --output preference_candidates.jsonl
```

Ask DeepSeek to turn candidates into preference pairs:

```bash
$env:DEEPSEEK_API_KEY="your_api_key"
python score_preferences.py --input preference_candidates.jsonl --output preferences.jsonl --mode deepseek --domain casual --append --dedupe --deepseek_parallel 4
```

If DeepSeek rejects every candidate for a prompt, it can now synthesize a better
`ideal_answer`; the script writes that answer as `chosen` and the worst local
candidate as `rejected`, so the next DPO step still has a useful learning signal.
Add `--no_synthetic_chosen` to disable this behavior.

Run a minimal DPO pass from the current checkpoint:

```bash
python train_dpo.py --preferences preferences.jsonl --checkpoint checkpoints_gravity120/model.pt --tokenizer checkpoints_gravity120/tokenizer.json --checkpoint_dir checkpoints_dpo_step1 --max_iters 20 --batch_size 1 --reference_free --log_every 1 --device cuda
```

Repeat this loop in small steps: generate candidates, let DeepSeek judge, run a
short DPO pass, evaluate, then continue from the new checkpoint. Use
`--mode manual` for human scoring or `--mode heuristic` for an offline bootstrap
pass when no API key is available.

## Online Reinforcement Learning

After supervised training and DPO, use the lightweight RL loop to improve
problem-solving behavior from the model's own sampled answers. The script
scores each rollout with a task reward based on target-answer overlap,
actionability, length, and quality gates. The update uses several sampled
answers per prompt, group-normalized advantages, PPO-style ratio clipping, and
a frozen reference checkpoint to keep the policy from drifting too far in one
run.

Smoke-test the setup first:

```bash
python train_rl.py --checkpoint checkpoints_dpo_casual_step8_final/model.pt --tokenizer checkpoints_dpo_casual_step8_final/tokenizer.json --max_examples 100 --batch_size 1 --dry_run --device cpu
```

Run a small RL pass:

```bash
python train_rl.py --checkpoint checkpoints_dpo_casual_step8_final/model.pt --tokenizer checkpoints_dpo_casual_step8_final/tokenizer.json --checkpoint_dir checkpoints_rl_step1 --max_examples 1000 --max_iters 50 --batch_size 2 --rollouts_per_prompt 2 --learning_rate 2e-6 --kl_coef 0.03 --ppo_clip 0.2 --device cuda
```

Then evaluate `checkpoints_rl_step1` with `eval_prompts.py` and keep iterating in
small runs. If outputs become verbose, repetitive, or drift away from the
expected answer style, lower `--learning_rate`, increase `--kl_coef`, or reduce
`--max_iters`.

The main generated corpus is `corpus/qa_10k.txt`: 8,000 Q/A records stored as
exactly 16,000 lines. `corpus/dialogues_clean.txt` adds clean multi-turn data,
and `corpus/priority_qa.txt` provides high-priority fallback answers for common
short prompts.

## Model Presets

Default training now targets a denser daily-use Transformer at about 304M
parameters:

```bash
python train.py --dry_run --device cpu
```

Expected default size with `tiktoken`:

```text
gravity-daily: 304,307,200 trainable parameters
```

Train the dense daily checkpoint on CUDA:

```bash
python train.py --model_preset gravity-daily --checkpoint_dir checkpoints_daily --batch_size 1 --grad_accum_steps 16 --max_iters 5000 --device cuda --grad_checkpointing
```

Available presets:

- `tiny`: quick CPU smoke tests
- `small`: old small-model scale
- `gravity-30m`: low-cost tiktoken training
- `gravity-60m`: mid-size training target
- `gravity-120m`: compatibility 120M architecture with GQA
- `gravity-120m-long`: 120M GQA layout with NTK-style RoPE scaling for longer
  context fine-tuning
- `gravity-daily`: default 304M dense Transformer for daily-use training
- `gravity-daily-long`: dense daily layout with NTK-style RoPE scaling
- `gravity-700m-dense`: larger dense target for stronger generality when
  hardware allows it

## Long Context

Gravity now follows the more mature small-LLM context pattern: RoPE can be
scaled for longer windows, checkpoints persist that architecture setting, and
generation packs complete recent turns by token budget instead of blindly
cutting the raw prompt.

Smoke-test the dense long-context architecture:

```bash
python train.py --model_preset gravity-daily-long --seq_len 2048 --dry_run --device cpu
```

Fine-tune from an existing compatible checkpoint with a longer window:

```bash
python train.py --model_preset gravity-daily-long --resume_from checkpoints_daily/model.pt --checkpoint_dir checkpoints_longctx_step1 --seq_len 2048 --max_iters 200 --batch_size 1 --grad_accum_steps 16 --learning_rate 5e-5 --device cuda --grad_checkpointing
```

Use more conversation history at inference:

```bash
python generate.py --checkpoint checkpoints_longctx_step1/model.pt --tokenizer checkpoints_longctx_step1/tokenizer.json --chat --max_history 8 --max_prompt_tokens 1800
```

Evaluate whether a checkpoint uses recent context:

```bash
python eval_context_memory.py --checkpoint checkpoints_longctx_step2_context/model.pt --tokenizer checkpoints_longctx_step2_context/tokenizer.json --device cuda
```

Run fast architecture smoke tests:

```bash
python test_long_context.py
```

For new long-context training runs, prefer `--model_preset gravity-daily-long`.
For old checkpoints, continue from the checkpoint first, then extend context in
small fine-tuning steps so the model learns to use the larger window.

The architecture choices track common open LLM designs: LLaMA/Qwen-style
decoder-only blocks with RMSNorm, RoPE, SwiGLU and dense attention by default,
plus long-context RoPE scaling and token-budgeted chat history packing. GQA is
kept for small/legacy presets. Useful references:
[Mistral 7B](https://arxiv.org/abs/2310.06825) and
[Qwen2 Technical Report](https://arxiv.org/abs/2407.10671).

## Main 120M Checkpoint

The trained 120M checkpoint is saved in `checkpoints_gravity120`:

```bash
python generate.py --checkpoint checkpoints_gravity120/model.pt --tokenizer checkpoints_gravity120/tokenizer.json --ask "在吗"
python generate.py --checkpoint checkpoints_gravity120/model.pt --tokenizer checkpoints_gravity120/tokenizer.json --chat
```

It was trained from the previous-default `gravity-120m` preset for 800 steps on
the previous 10,000-line corpus. The final validation loss from that run was
`0.0248`.

## Continue Your Latest Checkpoint

Your existing trained checkpoint is `checkpoints_latest`. It is a 58M legacy
architecture checkpoint and can be resumed safely:

```bash
python train.py --resume_from checkpoints_latest/model.pt --checkpoint_dir checkpoints_latest_clean --max_iters 800 --eval_every 200 --eval_iters 4 --batch_size 16 --learning_rate 3e-4 --min_lr 3e-5 --warmup_steps 50 --device cuda --no_progress
```

The current improved continuation is saved in `checkpoints_latest_clean`.

## Generation

Use the 120M checkpoint:

```bash
python generate.py --checkpoint checkpoints_gravity120/model.pt --tokenizer checkpoints_gravity120/tokenizer.json --ask "在吗"
python generate.py --checkpoint checkpoints_gravity120/model.pt --tokenizer checkpoints_gravity120/tokenizer.json --chat
```

`generate.py` suppresses invalid UTF-8 tokens and uses an adaptive fallback when
model output is empty, repetitive, mixed-language garbage, or mojibake. The
fallback now checks intent and constraints first: emotional support, task
planning, learning/explanation, rewriting, safety, and "make it more flexible"
requests are reshaped instead of blindly copying the nearest corpus row.
For pure model debugging:

```bash
python generate.py --checkpoint checkpoints_gravity120/model.pt --tokenizer checkpoints_gravity120/tokenizer.json --ask "在吗" --no_retrieval_fallback
```
