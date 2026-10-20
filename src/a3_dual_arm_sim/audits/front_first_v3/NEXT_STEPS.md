# SUPERSEDED: 2026-10-19 user requests 2x5-only quick training

Active plan: ../quick_2x5_v1/NEXT_STEPS.md. Do not resume multitask training or run old six-cell evaluation. Historical plan follows.

# Authorized follow-up

User approved front-camera image bottom-first placement, complete-release data repair, small-batch gate, recollection, recovery examples, independent 40k training, and paired new-seed evaluation. Keep H=8 and 8D left arm; no runtime action correction rules.

Active jobs: 2x5 collect20 (26232), pipeline 20-to-100 (93335); 2x10 collect20 (61321), pipeline (56887). Both smoke tests passed: 2x5 469 steps/10 cookies; 2x10 n=5 1066 steps/20 cookies, DONE and all released. continue_collection.py owns pipeline.lock and waits on collection file lock; it validates 20 episodes then extends to 100 only on pass. Check pipeline_status.json / pipeline_status_2x10.json, gate_20.json / gate_2x10_20.json, and collection logs before any launch. Each pipeline extends its task to 100 only when the gate passes. Never duplicate writers.

1. Verify smoke_2x10/seed_23001/result.json (n=5) and actual camera-column order [1,0]. Collect randomized first-grasp 2x10 data in separate cookie_2x10_frontfirst_v3, first 20 then 100, with --complete-demonstrations --diagnostic-dir; validate skipped grasps and all non-skipped releases, column order, retreat and stability. Do not reuse old 2x10 order.
2. Review failed attempts, both tasks phase coverage, target movement and final release. Add targeted recovery demonstrations for grasp misalignment/empty grasp/box-rim approach using existing dagger collection facilities only after checking their observation/action contract; preserve honest partial vs successful metadata and category counts.
3. Prepare new merged left8 dataset, preserving original v1 and full provenance. Run numeric/image contract audit. Start independent 40k training using old run hyperparameters and initial pretrained base, not 40k weights or optimizer resume. Save intermediate checkpoints for development selection.
4. Use new disjoint seeds for paired old/new model evaluation under common stricter released-and-stable evaluation, recording requested front-first order as a separate metric. Do not pass complete_demonstrations to a learned policy directly: that flag also requires expert DONE. Implement learned-policy evaluation completion separately.

Initial strict 2x5 smoke seed22001: 469 steps, DONE, 10/10, both reports released; target drift 0.272 mm. 23 focused tests passed. Current complete_demonstrations flags are opt-in so legacy benchmarks remain reproducible.
