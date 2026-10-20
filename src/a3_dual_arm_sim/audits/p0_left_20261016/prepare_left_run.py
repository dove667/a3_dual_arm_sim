"""Validate the 8-D projection and prepare, but do not launch, a 40k run."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from a3_dual_arm_sim.learning.training import train_smolvla
from a3_dual_arm_sim.paths import project_root


def main():
    root = project_root()
    source = root/'datasets/cookie_multi_v1'
    target = root/'datasets/cookie_multi_left8_v1'
    report_root = Path(__file__).resolve().parent
    count = 0
    for file in sorted((source/'data').rglob('*.parquet')):
        a = pq.read_table(file)
        b = pq.read_table(target/file.relative_to(source))
        assert a.num_rows == b.num_rows
        for key in a.column_names:
            if key in ('action', 'observation.state'):
                np.testing.assert_array_equal(np.asarray(a[key].to_pylist())[:, :8], np.asarray(b[key].to_pylist()))
            else:
                assert a[key].equals(b[key]), key
        count += a.num_rows
    stats_a = json.loads((source/'meta/stats.json').read_text())
    stats_b = json.loads((target/'meta/stats.json').read_text())
    for key in ('action', 'observation.state'):
        for stat, value in stats_b[key].items():
            original = stats_a[key][stat]
            assert value == (original[:8] if isinstance(original, list) and len(original)==16 else original)
    # Retain the prior run's batch, chunk length, tasks and sampling. Only the
    # requested left-arm projection and 20k -> 40k schedule are proposed here.
    os.environ['A3_SMOLVLA_SDPA'] = '1'
    result = train_smolvla(
        dataset_root=target, repo_id='local/cookie-multi-left8-v1',
        base_model='lerobot/smolvla_base',
        output_dir=root/'outputs/cookie_multi_left8_40k_v1/model',
        steps=40000, batch_size=32, seed=1000, device='cuda',
        lr=5e-5, save_freq=2000, num_workers=2, offline=True, dry_run=True,
    )
    config = result['policy_config']
    assert config['input_features']['observation.state']['shape'] == [8]
    assert config['output_features']['action']['shape'] == [8]
    assert config['scheduler_decay_steps'] == 40000
    result.update(status='prepared_not_started', launch_gate='attention numerical check requires resolution',
                  command_environment={'A3_SMOLVLA_SDPA': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                  projection_verified_frames=count, estimated_dataset_passes=40000*32/count,
                  changed_from_previous=['state/action 16D -> left 8D', 'training and LR decay 20000 -> 40000'],
                  source_info_sha256=hashlib.sha256((source/'meta/info.json').read_bytes()).hexdigest(),
                  left_info_sha256=hashlib.sha256((target/'meta/info.json').read_bytes()).hexdigest())
    (report_root/'left_training_plan.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:result[k] for k in ['status','projection_verified_frames','estimated_dataset_passes','launch_gate']}, indent=2))


if __name__ == '__main__':
    main()
