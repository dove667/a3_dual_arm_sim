# 30k SmolVLA basic diagnosis (2026-10-20)

Checkpoint: `outputs/cookie_2x5_frontfirst_basic_medium_left11_40k_resume_v1/model/checkpoints/030000/pretrained_model`; weights SHA256 starts `529ce69d40a925`. This is the 30k snapshot, not a 40k result. Inputs: front, left wrist, right wrist, left joint angles7 + gripper1 + world EEF xyz3; output left arm8. Training dataset: 200 full 2x5 demonstrations (100 basic,100 medium); the ongoing second-round collection is not in these weights.

Protocol: 2x5/basic/source-column1/training prompt, seeds13300–13304, inference seed13000, max500, n_action_steps8, flow steps25, CUDA, one worker, no action fixes. Each run JSON stores checkpoint and processor fingerprints. Diagnostic traces/videos and run JSON live in this folder. All five trace/video lengths match 500 frames. The initial sandbox CPU attempt was stopped before completion; only `seed_13300_cuda` and `seeds_13301_13304` count.

Results: 0/5 complete tasks and 0/50 final cookies; each episode timed out at500. In every episode, maximum observed cookie lift was under2mm, and no cookie lifted10mm. The left gripper did close (minimum applied opening roughly0.27), but physical lift did not follow. Two-finger simultaneous contact frames/longest consecutive run: seed13300 12/4; 13301 2/1; 13302 0/0; 13303 8/3; 13304 7/1. Thus the recurring failure is initial grasp stability, before second placement.

Matched expert under same basic seed13300 and official front-first complete-demonstration contract: 10/10 success in473 steps. At first closure (steps65–80), learned EEF mean XY [0.10012,0.24079]m, expert [0.09389,0.25081]m; XY separation11.8mm. Learned EEF kept both left finger contacts for at most4 consecutive frames and maximum cookie lift0.5mm; expert held both finger contacts116 consecutive frames and lifted the batch about81mm. This is a measured spatial/temporal difference, not proof of which input or training factor caused it. Initial wrist image shows the selected source cookies visible; no evidence yet that occlusion caused this failure.

Implication: Prioritize first-grasp pose and close/lift timing. Because no model episode reached a valid first placement, these runs provide no evidence about second-round pickup or placement quality. Do not use this 5-seed development sample as a population success-rate estimate. For next controlled test, compare 40k to 30k on paired basic seeds with identical processors and protocol, then inspect first-grasp displacement and sustained two-finger contact. If 40k still fails, add successful pose-correction and recovery data while retaining full-task demonstrations; independently audit whether model observations and expert target trajectories share the same state/action semantics and camera timing.

Reproduction from project root (set `MUJOCO_GL=egl`):
```
python examples/benchmark_cookie_batch.py --task 2x5 --profile basic --source-column 1 \
  --policy smolvla:outputs/cookie_2x5_frontfirst_basic_medium_left11_40k_resume_v1/model/checkpoints/030000/pretrained_model \
  --dataset-root datasets/cookie_2x5_frontfirst_basic_medium_left11_v1 \
  --repo-id local/cookie-2x5-frontfirst-basic-medium-left11-v1 \
  --device cuda --prompt-style training --episodes 5 --seed-start 13300 \
  --inference-seed 13000 --max-steps 500 --n-action-steps 8 --num-steps 25 \
  --workers 1 --diagnostic-dir outputs/basic30k_reproduce_diagnostics \
  --output outputs/basic30k_reproduce.json
```
