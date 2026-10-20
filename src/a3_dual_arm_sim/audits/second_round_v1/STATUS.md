# Second-round collection (2026-10-20)

User authorized targeted collection, not training on these new samples. Existing 40k resume and grid watcher remain independent.

Active production: datasets/cookie_second_round_medium100_v2 (project-root relative), seed start30000. Module a3_dual_arm_sim.data.collect_second_round_cli. Log /tmp/cookie_second_round_medium100_v2.log. Single writer; resume same command skips committed episodes after checking persisted metadata/data counts. Interrupted uncommitted attempt directories are retained and skipped, a new seed is used.

Quotas: normal40 (seeds30000+), offset30 (31000+), controlled empty recovery20 (32000+), first-stack variation10 (33000+). Profilemedium, sourcecolumn1, front-camera bottom-first. All prefixes physically performed by expert. This is controlled expert recovery data, NOT model-failure takeover/DAgger. Empty recovery covers empty closure above source, open, lift20mm, reapproach, grasp; does not claim partial-grasp/drop recovery. Offset5–12mm and yaw0.05–0.12rad at the second approach; first placement localx offset±1mm. Deliberate perturbation prefix is excluded from training suffix.

Pilot datasets/cookie_second_round_medium100_v1: normal25000, offset26000, empty27000 passed first attempt, first-stack offset±2mm passed28005 after five failures (28000–28004). All failures retained and excluded. Refined +1mm first-stack seed28000 passed in datasets/cookie_second_round_pilot_refined/seed_28000, which failed at+2mm. This is a pilot, not evidence of model improvement. Pilot samples not counted toward production100.

Every episode has full3-camera diagnostic videos, actual cookie poses/contacts, actions/states, and a separately recorded contiguous suffix. Writer closes/finalizes per episode before COMMITTED.json is atomically written. Gates check task10/10 stable20steps, expertDONE, two five-cookie actual lifts+release reports, first-stack5 at suffix start, perturbation realized, Parquet frame counts, saved applied-action equality against trace suffix, three diagnostic videos decoded count matching full trace. Each successful shard converted to left11/action8. Model input unchanged; privileged state stays in diagnostic/sidecar.

Production merges only after exact40/30/20/10 valid counts, recomputes stats, repairs derived episode metadata indices, audits merged counts. COMPLETE.json means100 shards finished; TRAINING_DATA_READY.json means merged training_left11 dataset audited. Do not start new training automatically. Never resume pilotrootv1 with current code (protocol changed); productionv2 hashes its collector version. Stop on unexpected worker errors and inspect worker.log.

Verified: four pilot categories plus refined first-stack successful; source py_compile; suffix gating test; active collector rejects second lock; git diff --check. Need after production: verify resume skips saved shards, inspect per-category distribution and critical visual frames, and report actual100 dataset audit.
