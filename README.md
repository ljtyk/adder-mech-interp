# What the Adder Learned

A research log on how a small length-generalizing addition transformer works inside. The post is `index.html` (open it in a browser, or see the GitHub Pages site). It is a companion to [Carry the One](https://github.com/ljtyk/carry-the-one), which built the model, and to [Why Rotary Embeddings Stop Extrapolating](https://github.com/ljtyk/rope-aliasing).

**Short version.** The model reads two numbers digit by digit, writes each digit into its own slot, repeats one transformer block (the *combiner*) 24 times over those slots, and reads the sum back out. Recording its state after every round, probing what each digit position knows, tracing attention and swapping states between runs shows a fixed order of work: each place computes its own digit sum and carry-out within two rounds, a carry then travels one place at a time (about 1.85 rounds per place along a run of nines; no sign of carry-lookahead), and the answer slowly settles. The state never reaches a fixed point, so v22's accuracy decays if it is run past 24 rounds. Training with a random number of rounds (v23) removes the decay within the trained lengths, and v33 (80 rounds, operands up to 64 digits) is at 99.7% or better on every fixed test set and exact on the 24–26 digit one. A final length test shows the limit is the training range, not the computation: v33 is at 99.5% or better through 64 digits, 74.5% at 80, and running it 60 extra rounds changes nothing.

## Layout

- `index.html`: the post. `build_post.py` regenerates it from `data/`, and checks a number of the claims in the text against the data as it builds (it fails if they stop being true).
- `data/`: the result files every number and chart in the post comes from.
  - `v22_dynamics.json`, `v23_dynamics.json`, `v33_dynamics.json`: accuracy at every round, how long answers take, size and direction of the state's change per round, crafted carry chains (`probe_combiner_dynamics.py`).
  - `v22_carry_probes.json`: linear probes on a place token by round (`probe_combiner_carry.py`).
  - `v22_attention.json`: attention by relative place and scratch-token traffic (`probe_combiner_attn.py`).
  - `v22_patching.json`: state swaps and the carry's speed (`probe_combiner_patch.py`).
  - `v22_scratch_tokens.json`: scratch-token ablations, attention and probes (`probe_thinking_tokens.py`).
  - `v22_eval.json`, `v23_eval.json`, `v33_eval.json`: accuracy on the fixed test sets (`eval_additive.py`).
  - `v33_limits_r80.json`, `v33_limits_r140.json`: the length test at 80 and 140 rounds (`eval_additive_limits.py`).
  - `logs/`: the raw run logs. A few numbers (sample sizes, scratch-token probe baselines, the extra state-swap rounds) are read from these.
  - `base_style.css`: the page style shared with the other two posts.
- `code/nanoGPT/`: model, training, evaluation and probes (PyTorch).
  - `model.py`: all model variants. `AdditiveComposedGPT` is the model in the post. Three inert hooks were added for this study: `_combiner_trace` (record the state after every round), `_combiner_hook` (rewrite the state between rounds), and `CausalSelfAttention._attn_store` (record attention weights). They do nothing unless a caller sets them.
  - `train.py`, `configurator.py`: training. `config/train_addition_additive_v22.py`, `_v23.py`, `_v33.py` are the three runs in the post. v23 and v33 warm-start from v22's checkpoint.
  - `eval_additive.py`: accuracy on the fixed test sets. `eval_additive_limits.py`: the length test, optionally with extra rounds.
  - `probe_combiner_dynamics.py`, `probe_combiner_carry.py`, `probe_combiner_attn.py`, `probe_combiner_patch.py`, `probe_thinking_tokens.py`: the five probes.
  - `data/addition/`: the problem generator and the fixed test sets; `data/addition_padded/prepare.py` writes the vocabulary file the probes read.

## Reproducing it

Requirements: Python 3.10+ and PyTorch 2.x (plus NumPy and Matplotlib for the probes). Trained checkpoints are not included. From `code/nanoGPT/`:

```
python data/addition/generate.py            # regenerates the test sets byte for byte (they are also included)
python data/addition/prepare.py
python data/addition_padded/prepare.py      # writes meta.pkl, which the probes read

python train.py config/train_addition_additive_v22.py     # v22: well under an hour on a laptop RTX 4080
python train.py config/train_addition_additive_v23.py     # warm-starts from out-addition-additive-v22/ckpt.pt
python train.py config/train_addition_additive_v33.py     # warm-starts from v22; a few hours

python probe_combiner_dynamics.py out-addition-additive-v22 ckpt.pt
python probe_combiner_carry.py    out-addition-additive-v22 ckpt.pt
python probe_combiner_attn.py     out-addition-additive-v22 ckpt.pt
python probe_combiner_patch.py    out-addition-additive-v22 ckpt.pt
python probe_thinking_tokens.py   out-addition-additive-v22 ckpt.pt
python eval_additive.py --out_dir=out-addition-additive-v33 --ckpt_name=ckpt_15000.pt --device=cpu
python eval_additive_limits.py --out_dir=out-addition-additive-v33 --ckpt_name=ckpt_15000.pt --device=cpu
python eval_additive_limits.py --out_dir=out-addition-additive-v33 --ckpt_name=ckpt_15000.pt --device=cpu --extra_rounds=60
```

Each probe writes a JSON file (and some plots) into the run directory; the files in `data/` are those, renamed. Initialization is seeded, but training problems are generated on the fly without a fixed seed (and GPU kernels are not deterministic), so a retrained checkpoint will not match the published one digit for digit.

## Notes on the data

- **Round numbering.** In the dynamics, carry-probe and patching files, index *k* of a per-round array is the state after *k* rounds (index 0 is before the first). In `v22_attention.json` and the attention part of `v22_scratch_tokens.json`, index 0 is the first application of the block, so the post labels those rounds from 1.
- **Mislabeled keys.** In the `*_dynamics.json` files the keys `acc_at_24` and `acc_at_64` mean "at `repeat_max`" and "at `extended_rounds`" (24 and 64 for v22 and v23; 80 and 120 for v33), and the run log prints them with the same labels. The post computes everything from the full per-round `curve` arrays.
- **Which checkpoints.** v22: `ckpt.pt` (iteration 9,500). v23: `ckpt.pt` (iteration 10,500). v33: `ckpt_15000.pt`; the training script's own "best" checkpoint for v33 stopped at iteration 3000 (it gates on a clipped validation loss that did not track accuracy) and is not used.
- **v22 evaluation.** `v22_eval.json` is the original GPU run. `logs/v22_eval.log` is a CPU re-run in 32-bit floats; the numbers match.
- **Sample sizes.** The round-by-round probe uses 150 problems per length set (60 for v33, whose rounds are twice as many and 128 slots wide), so those curves differ from the full test sets by a few points.

The model code follows [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT, see `code/nanoGPT/LICENSE`).

Written with Claude (Anthropic) as research assistant.
