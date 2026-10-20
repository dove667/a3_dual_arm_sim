# ModelScope dataset alignment audit

Revision: ce3f95672280b250a6f9b3421fdc7c4490b9f845
Source: https://www.modelscope.cn/datasets/yunsx114/a3-dual-arm-cookie-main-aligned

## Verdict
Structurally compatible A3 data, but not directly compatible with the current front-camera-bottom-first training target. Do not merge unfiltered. No full physics replay or video-pixel alignment certification was performed.

## Evidence
- Remote info: 1,000 episodes, 425,354 frames, LeRobot v3, 20 Hz, three 256x256 cameras, state/action 16D joint positions. Current left8 projection is technically possible after semantics validation.
- Downloaded numeric shard file-000.parquet: 297,747 frames, 700 complete episodes (0-699), SHA256 matches published manifest. Metadata parquet also matches manifest.
- Heuristic gripper close/open event analysis: all 350 sampled full/perturb episodes have two releases, first EEF world x around 0.06698, second around 0.123. With target x=0.095 and slot x=+/-0.028 this is target column order [0,1]. All 350 sampled prefill variants only execute the positive-x-column release. Current new order is [1,0]. This is an incompatible training target, not a tensor conversion issue.
- All eight explicitly provided camera parameters match current effective cookie_batch config. Eighteen explicit geometry/contact/gripper fields match. This is config agreement only; generating simulator revision and runtime apply_scene implementation are not included.
- Target config z differs: donor 0.74898 vs current configured 0.753. Runtime reset/settling can alter both, so this is not proof of a 4.02mm actual mismatch. Donor observed release EEF z is about 0.7597; replay is required before trusting current-scene action alignment.
- README states pinned source/target positions and home pose, only target yaw sigma=0.005rad; current basic rollout randomizes boxes by 10mm and cookies. Fixed-layout coverage cannot substitute for current randomized evaluation.
- Success name says upright but require_upright=false and require_exact_slots=false. 40-step release/settling does not establish our upright/slot success criterion.
- Modes are seed modulo four: 250 full, 250 prefill, 250 perturb, 250 prefill_perturb. Prefill starts from teleported first batch; perturb alters first placement. These are useful disturbance scenarios, not demonstrated four-cookie-grasp or wall-collision recovery.
- Prompt differs from our short training prompt. Published sidecar uses episode_metadata.jsonl and different task identifier; existing audit/merger needs an explicit adapter.
- Mode/shard grouping requires stratified seed split. Donor normalization must not replace current checkpoint normalization; an accepted merged training split needs newly computed training-only stats.
- Root summary placements_randomized=true disagrees with README pinned-layout description; the summary cannot alone certify reset distribution.

## Recommended use
Prefer porting its prefill/perturb collection design into the current environment and recollecting with [1,0] placement order, current cameras/randomization, and strict upright/released/stable completion. For direct use first replay representative episodes in the current environment, measure final strict success and action/state synchronization, inspect all-camera imagery, and isolate any compatible trajectory segments. Do not relabel or swap only actions to reverse order; observations and robot configurations would no longer match.

Any subsequent augmentation experiment should retain 2x10 and normal front-first data, use controlled source sampling, and compare an equal-budget baseline against augmented data on disjoint paired seeds. No training or dataset merge was launched by this audit.
