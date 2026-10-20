# SmolVLA basic checkpoint and inference-parameter audit (2026-10-20)

## Scope and protocol

Task 2x5/basic/source-column1/training prompt, episode seeds13300–13304, fixed inference seed13000, CUDA, one worker, default simulation action rules disabled, action execution horizon8, flow25, max500 unless a probe explicitly changes one of these. The common dataset is `datasets/cookie_2x5_frontfirst_basic_medium_left11_v1` (200 full demonstrations, basic100 + medium100), state11 (left joint7 + gripper + world EEF xyz), action8, three cameras, chunk50. Newly collected second-round examples were not used in these checkpoints. Reports contain weight and processor fingerprints; every completed episode's trace length and all three video lengths were verified (43 episodes).

Checkpoints 20k came from the original20k run; 22k–32k came from exact optimizer/data-order resume on the same dataset, batch32. Training optimizer peak5e-5, cosine floor5e-6 reached20k; all resumed checkpoints remained at5e-6. Training loss stayed around0.01 after20k. Configurations preserve `freeze_vision_encoder=true`, `train_expert_only=true`, `train_state_proj=true`, num_steps25. Normalizer tensor values were identical at20k and24k despite different binary file hashes. Model weight keys matched and frozen weights were unchanged; no evidence of an action/state dimension or normalizer contract change in the compared checkpoints.

## Paired five-seed results

| checkpoint | complete success | final cookies / 50 | episodes lifting >=5 cookies at least10mm | episodes reaching >=5 target cookies transiently |
|---|---:|---:|---:|---:|
|20k|0/5|11|5/5|3/5|
|22k|0/5|24|5/5|5/5|
|24k|0/5|0|1/5|0/5|
|28k|0/5|10|2/5|2/5|
|30k|0/5|0|0/5|0/5|
|32k|0/5|5|1/5|1/5|

22k is the best **development candidate** by the secondary score and first-grasp consistency, but no checkpoint achieved a complete task. Later checkpoints do not monotonically improve. For seed13300, the 20k expert-grasp first-close EEF y was about0.2503m, 24k about0.2435m, 30k about0.2408m, while the matched physical expert was about0.2508m. The 30k episode could not lift any cookie; the expert completed10/10. For 20k seed13300, the first batch entered the target box transiently at step170 but slid/moved out; learned EEF x about0.141m versus expert about0.130m near first release. These are measured trajectory differences, not proof of a particular visual or training cause.

## Parameter probes

| checkpoint/probe | seeds | complete success | final cookies | >=5-cookie-lift episodes |
|---|---:|---:|---:|---:|
|20k, H8, flow25, cap500|13300–13304|0/5|11/50|5/5|
|20k, H4, flow25, cap500|13300–13302|0/3|0/30|1/3|
|20k, H8, flow10, cap500|13300–13304|0/5|20/50|4/5|
|22k, H8, flow25, cap500|13300–13302|0/3|14/30|3/3|
|22k, H8, flow10, cap500|13300–13302|0/3|10/30|2/3|
|22k, H8, flow25, cap650|13300–13302|0/3|20/30|3/3|
|30k, H8, flow25, cap500|13300–13302|0/3|0/30|0/3|
|30k, H4, flow25, cap500|13300–13302|0/3|0/30|0/3|

H4 was worse for both tested checkpoints; retain H8. Flow10 raised final counts for20k but worsened22k on the same three seeds and never produced a complete success; do not change the default from25 on these data. Cap650 let two22k episodes reach10/10 contained cookies but both still failed exact slot occupancy. In seed13300 the final slot occupancy was `[5,-1,6,0,7,1,8,2,9,3]`: one required slot remained empty despite ten cookies in the box, consistent with the observed overlapping second placement. Cap650 also produced0/10 on seed13301 where the cap500 run had5/10, so more time is not a monotonic benefit.

## Repeatability boundary

Repeating the **same** 22k/H8/flow25/max500/seed13300 configuration produced final scores4/10 and10/10; both failed exact-slot success. The first different applied action appeared at step16. On one frozen observation, two reset/predict sequences of24 actions were identical. In two same-seed closed-loop100-step runs, raw camera frames diverged before the first action difference: at most1/255 in sparse pixels in the first16 frames, with mean absolute differences <=1.2e-5; action differences emerged at step24 and robot state differences at step25. Thus the fixed-input policy sampler is repeatable, while tiny camera/render or other closed-loop changes can precede and amplify trajectory divergence. The raw camera difference is observed; its precise renderer cause and the causal sensitivity are not yet isolated. Paired seed results are descriptive small-sample development evidence, not a statistically secure checkpoint ranking. The max500/max650 score difference also cannot be attributed solely to the step cap because prefix actions differed on repeat.

## Recommended next action

Keep 22k as a provisional development checkpoint for data iteration; retain20k as a comparison. Do not deploy a new default based on these samples: all complete-task success rates were0. Keep H8 and flow25 for the next comparison. A larger cap can be used for diagnosis, but task acceptance must require correct one-to-one slots and stable hold, not ten cookies counted inside the box. Prioritize second-stack release pose/column placement and recovery examples for22k, while keeping full-task first-grasp demonstrations in any new training mix. Before claiming new-data gains, use independent seeds (e.g.13400–13419), repeated trials if practical, and report complete success plus grasp, lift, transient placement and final exact-slot counts. Do not select solely by training loss. The ongoing40k run and its already-armed grid evaluation should be assessed when complete; no new training or default setting was changed by this audit.

Data: `comparison.json`, per-run `*.json`, per-episode `trace.jsonl` and three MP4s. Reproduction scripts: `run_saved_checkpoints.sh`, `run_later_and_probes.sh`, `run_followups.sh`, `run_22k_flow10.sh`; comparator `compare.py`. Fixed-input and repeated-closed-loop checks: `fixed_observation_repeat.json`, `closed_loop_repeat.json`.
