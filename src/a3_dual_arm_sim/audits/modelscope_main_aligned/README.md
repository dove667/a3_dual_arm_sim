---
license: cc-by-4.0
task_categories:
- robotics
tags:
- lerobot
- mujoco
- manipulation
- dual-arm
- imitation-learning
- sim
size_categories:
- 1K<n<10K
---

# A3 dual-arm cookie transfer — the benchmark arrangement, pinned

1000 expert demonstrations of a two-batch pick-and-place task on a simulated dual-arm robot
(A3, 14 joints, position-controlled), recorded on **the arrangement the upstream
`CookieBatchBenchmark` ships with**, held fixed from episode to episode.

This is the sibling of `cookie_enhanced_1k` (`a3_b2`): same task, same prompt, same contract,
same features, same four conditions — a *different arrangement*.  `a3_b2` randomises both
boxes, the arm's home pose and the camera block per episode; this set pins all of them to the
benchmark's own scene and varies only the target box's yaw, σ = 0.005 rad.

**Why a second arrangement is worth a second dataset.**  A policy trained on `a3_b2` scored
5.35 cookies on its own layout and **0/20** on the benchmark's.  The difference was not
physics, not the judge, not the cameras (all three were verified identical); it was the
arrangement — a 23 mm bin offset is enough to turn a 5/10 policy into a 0/20 one.  Fine-tuning
the same checkpoint on these 1000 episodes for 4000 steps took that benchmark score from
**0.00 to 7.55** (20 seeds, mean cookies out of 10) without losing the old layout (5.35 →
6.25).  So the arrangement is a real axis of generalisation, not noise.

**Task prompt** (part of the task; use it verbatim):

> grasp five yellow blocks together from the source bin, insert them upright into the marked
> slots of the 2 by 5 box, then return to the source bin, grasp five more, and fill all ten
> slots

**Success contract**: `released_10_upright_settled_contained_70_remain` — ten Cookies
released, settled and contained in the target box, seventy left in the source bin, held for 40
steps.  Uprightness is **not** required at acceptance (`require_upright: false`,
`require_exact_slots: false`): the scene grades a Cookie whose low end rests on the box floor
as placed, the same relaxation the second-batch collection used.

## Quick start

```python
# Take the *whole* snapshot: if a piece is missing locally (the videos, say), lerobot goes
# looking for it on the Hugging Face Hub, which a machine without access to huggingface.co
# cannot do -- and the failure is a connection error, not a missing file.
from modelscope import snapshot_download
root = snapshot_download("<owner>/a3-dual-arm-cookie-main-aligned", repo_type="dataset")

import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# `root` is passed explicitly so the dataset is read from the downloaded directory rather than
# looked up by repository id -- the id is only a cache key.
dataset = LeRobotDataset("local/a3_b2_aligned", root=root, video_backend="pyav")

# The four conditions are not a field in the dataset.  They are `seed % 4`, and the seed lives
# in the sidecar shipped next to the data -- `meta/episodes/*.parquet` carries no seed.
MODES = ["full", "prefill", "perturb", "prefill_perturb"]
sidecar = [json.loads(line) for line in open(f"{root}/episode_metadata.jsonl")]
mode_of = {r["episode_index"]: MODES[r["seed"] % 4] for r in sidecar}

full_only = [i for i, mode in mode_of.items() if mode == "full"]
```

`examples/load_and_filter_by_mode.py` is the same thing with the counts printed, and
`examples/train_pi05_example.sh` is the recipe these episodes were fine-tuned with.

## The four conditions

| mode | episodes | frames (min/mean/max) | what the episode contains |
|---|---|---|---|
| `full` | 250 | 548 / 548.7 / 550 | the whole task: empty box, both batches performed |
| `prefill` | 250 | 302 / 302.0 / 303 | **starts at the second batch**: five Cookies already in the box, imperfectly placed |
| `perturb` | 250 | 548 / 548.7 / 550 | the whole task, and the five just placed are nudged at the moment of release |
| `prefill_perturb` | 250 | 302 / 302.0 / 302 | `prefill`, with wider imperfections |

Two things to notice about those frame counts, both consequences of pinning the arrangement:
they are far tighter than `a3_b2`'s (530–874 for `full`), because the expert's path is nearly
the same every time; and they are *bimodal* — 548 or 302, nothing in between — because the
`prefill` variants teleport the first batch into the box instead of performing it.

The imperfection model is `a3_b2`'s, unchanged: in `prefill`, **one coherent lean for the
whole group**, uniform in `[0°, 4°]` (`prefill_perturb`: `[0°, 6°]`) at a drawn azimuth, yaw in
±1° (±1.5°), per-Cookie position jitter ±0.2 mm (±0.3 mm).  `perturb` re-applies a coherent
lean of up to 3° to the five just released.

## What one step contains

| key | type | shape |
|---|---|---|
| `observation.images.front` | video | 256 × 256 × 3 |
| `observation.images.left_wrist` | video | 256 × 256 × 3 |
| `observation.images.right_wrist` | video | 256 × 256 × 3 |
| `observation.state` | float32 | 16 |
| `observation.velocity` | float32 | 16 |
| `observation.eef_pose` | float32 | 14 |
| `observation.force` | float32 | 18 |
| `action` | float32 | 16 |

* `state` and `action` are **joint positions in radians**, 16 wide (7 joints + gripper per arm)
  in A3's own joint order.  They are not Panda, not LIBERO, and not Cartesian deltas.  The
  `action` is the joint target the expert commanded at that step; the arm is
  position-controlled, so this is a target rather than a torque.
* 425,354 frames at **20 fps** (dt = 0.05 s), from a 1000 Hz physics loop (50 substeps per
  control step).
* The videos are **AV1** (`libsvtav1`): decoding needs a build of pyav with an AV1 decoder.
* The images are the **unmirrored** convention.  Every checkpoint fine-tuned on LIBERO expects
  the horizontally mirrored view of the same scene — that flip is a switch on the environment
  side (`A3_IMAGE_MIRROR`), not something baked into these pixels.

## How it was collected

* After every reset the scene is rewritten onto a dump of the benchmark's own scene
  (`main_alignment.apply_scene`): both boxes, all 80 Cookies, the arm's home joints and the
  three cameras.  The **only** variation across the 1000 episodes is the target box's yaw,
  drawn with σ = 0.005 rad from the episode's seed.  Cookies are not jittered, and the arm's
  home pose is pinned — a 0.01 rad home-pose shift alone is enough to make the scripted expert
  fail.
* The controller is the scripted expert (`CookieCollectionPolicy`).  An episode was kept only
  if the run finished with the contract satisfied: **1000 attempts → 1000 accepted**, seeds
  2000–2999, all with `success: true`.
* `scene_config.yaml` is the scene file the collection ran on; `collection_summary.json` is the
  collection's own record (task, contract, prompt, arrangement, acceptance counts).
* Two gates, because "aligned in process" and "aligned in the data" are different claims: the
  in-process diff against the benchmark's dump reads **0.000 mm / 0.000 rad** on both boxes,
  all 80 Cookies, all 18 joints and the camera world poses; the first frame of the recorded
  episodes differs from the benchmark's own render by **2.25 / 3.34 / 1.48** mean absolute
  difference of 255 (front / left wrist / right wrist).  The frame gate is the larger number
  because it carries the 0.005 rad yaw draw and the expert's own first step.

## Known caveats

1. **No `mode` field, and no seed in `meta/episodes/`.**  The condition is recoverable only
   from `episode_metadata.jsonl` (which carries `seed` per episode) as `modes[seed % 4]`.
2. **The episodes are ordered in blocks of 50 by condition, five times over.**  The set was
   collected in 20 shards and each shard contributed its episodes grouped by mode, so along
   `episode_index` the sequence is 50 `full`, 50 `prefill`, 50 `perturb`, 50
   `prefill_perturb`, repeated — and every episode in a block shares the shard it came from.
   A sequential split therefore puts whole conditions, and whole shards, on one side.
   **Shuffle, or split by seed**, before measuring generalisation.
3. **The videos are AV1, and the decoder is a real dependency.**  lerobot's default video
   backend is torchcodec, which needs a matching FFmpeg build; where it is missing the failure
   reads as a missing shared library (`libavutil.so.57`) rather than as a codec problem.  Pass
   `video_backend="pyav"` (or `--dataset.video_backend=pyav` to `lerobot-train`).
4. **Normalise with this set's statistics.**  Its `observation.state` statistics differ from
   `a3_b2`'s by 0.083 rad on dimension 0 and 0.113 against 0.173 on that dimension's std —
   about 0.7 σ.  A policy fine-tuned here expects these statistics; feeding it the sibling
   set's is a silent input shift.
5. **A single number means nothing without the arrangement.**  Read scores as distributions
   over many seeds with the scene held fixed.  On this arrangement the fine-tuned policy reads
   {5: 9, 9: 4, 10: 7} over 20 seeds — a policy can plateau at exactly 5 (one batch) for many
   seeds and jump to 10 when the second batch goes; the mean hides which happened.

## Provenance

Collected with the simulator in the `a3_dual_arm_sim` repository (task
`a3_cookie_batch_second_batch_aligned`, scene file `configs/cookie_batch_main_aligned.yaml`,
included here as `scene_config.yaml`), on the arrangement of that repository's upstream
`CookieBatchBenchmark`.  The fine-tuning runs are not part of this repository.

## License

The data is released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**;
the full legal text is in `LICENSE`.  The simulator and scripts that produced it are released
under Apache-2.0 in their own repository.  If you use this dataset, cite the repository
(`a3_dual_arm_sim`) and its commit.
