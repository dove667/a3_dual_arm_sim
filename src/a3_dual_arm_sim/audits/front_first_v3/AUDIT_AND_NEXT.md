# Front-first data and training audit

{
  "accepted_data": {
    "2x5": 100,
    "2x10": 100,
    "frames_total": 150598,
    "left8_framewise_equal": true,
    "camera_video_keys": [
      "front",
      "left_wrist",
      "right_wrist"
    ]
  },
  "failed_attempt_categories": {
    "2x5": {
      "source_collateral": 3,
      "grasp_or_phase_timeout": 0,
      "unstable_release": 0
    },
    "2x10": {
      "source_collateral": 1,
      "grasp_or_phase_timeout": 4,
      "unstable_release": 1
    }
  },
  "recovery_compatibility": {
    "existing_dagger_collector": "2x5 only; fixed legacy task prompt/dataset path and old completion contract; no front-first release validation",
    "existing_manual_recovery": "one interactive episode, marked review_required and training_eligible false",
    "decision": "Do not mix old recovery files into the new dataset. Existing nine failed attempts do not contain a validated complete expert recovery continuation; train clean baseline first and implement task-specific recovery collection separately."
  },
  "training": {
    "split": "train_split.json",
    "dry_run": "train_dry_run.json",
    "new_output": "outputs/cookie_multi_frontfirst_left8_40k_v3/model",
    "base": "lerobot/smolvla_base",
    "steps": 40000,
    "execution_horizon_eval": 8,
    "flow_steps_eval": 25
  },
  "source_hashes": {
    "2x5": {
      "metadata_sha256": "55da88e118c33b3d1b34af352b80225eadf8533c2411050066400df2c696d2d3",
      "summary_sha256": "f761cd1ab2567278cc833bbd29bd06599850e0f9e3dbec748e24a91646ea665f"
    },
    "2x10": {
      "metadata_sha256": "c0bfaad73b58ee4649a12e4932e5d7a156e92ab6b06dcfc63a135f21dbd1bf16",
      "summary_sha256": "e146a1b9dcc62a66ae680da2eb0fb77fd1445aa3efe3dd22ace863cbc35491c6"
    }
  }
}
