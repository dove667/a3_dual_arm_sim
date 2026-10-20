# SUPERSEDED by user: 2x5 medium100, batch32, 20000 steps
Current known 2x5 datasets are basic, not medium. Do not restart basic training. Await dataset location or resolve collection requirement. All100 intended for training, no inferred80/20 split.

# Active plan, 2026-10-19 user update
Only local cookie_2x5_frontfirst_v3 (front camera lower stack first); no 2x10 or external data.
Independent left8 projection: 80 train / 20 holdout episodes. Normalization from train only.
Fresh original SmolVLA base, batch64, 4000 updates, lr5e-5, seed1000, save every1000. Verify CUDA memory on actual updates; if OOM, record failure and use batch48 in a separate run.
Old multitask v4 stopped at observed37551; preserved036000 checkpoint; do not resume or launch old six-cell evaluation.
After checkpoints exist: quick development evaluation only2x5 basic/source-column1/training prompt, max500, H8, flow25, inference seed13000. Reproduce seed13000 and use fixed development seeds13150-13154. Track full completion, final count, first/second grasp and placement. These are development checks, not accepted performance gains. Do not automatically extend training beyond4000 or start new data collection.

Batch64 failed with CUDA OOM on first update. Active replacement: batch48, outputs/cookie_2x5_frontfirst_left8_4k_b48_v1, launch_b48.sh. All other settings unchanged.
