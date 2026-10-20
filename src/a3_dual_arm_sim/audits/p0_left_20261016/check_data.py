"""Read-only P0 merge correspondence audit; writes only the requested report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from a3_dual_arm_sim.data.audit import audit_training_dataset


def rows(root, folder):
    return [r for p in sorted((root / folder).rglob('*.parquet')) for r in pq.read_table(p).to_pylist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    merged = root / 'cookie_multi_v1'
    sources = [root / n for n in ('cookie_2x5_v2', 'cookie_2x10_v2')]
    report = {'status': 'running', 'scope': 'source-to-merge correspondence; no simulator replay'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report, indent=2))
    save()
    report['contracts'] = [audit_training_dataset(p, repo_id='local/p0') for p in sources + [merged]]
    target = rows(merged, 'data')
    target_tasks = {r['task_index']: r['task'] for r in pq.read_table(merged/'meta/tasks.parquet').to_pylist()}
    meps = sorted(rows(merged, 'meta/episodes'), key=lambda r: r['episode_index'])
    samples = []
    offset = ep_offset = 0
    fields = ['observation.state', 'action', 'observation.velocity', 'observation.eef_pose', 'observation.force', 'timestamp', 'frame_index']
    for source in sources:
        original = rows(source, 'data')
        tasks = {r['task_index']: r['task'] for r in pq.read_table(source/'meta/tasks.parquet').to_pylist()}
        eps = sorted(rows(source, 'meta/episodes'), key=lambda r: r['episode_index'])
        for i, a in enumerate(original):
            b = target[offset+i]
            for key in fields:
                assert np.array_equal(a[key], b[key]), (str(source), i, key)
                assert np.isfinite(a[key]).all(), (i, key, 'nonfinite')
            assert b['index'] == offset+i
            assert b['episode_index'] == a['episode_index']+ep_offset
            assert tasks[a['task_index']] == target_tasks[b['task_index']]
        for ep in eps:
            dst = meps[ep['episode_index']+ep_offset]
            assert dst['length'] == ep['length']
            assert dst['tasks'] == ep['tasks']
            assert dst['dataset_from_index'] == ep['dataset_from_index']+offset
            for j in sorted({0, 1, ep['length']//2, ep['length']-2, ep['length']-1}):
                samples.append((str(source), ep['dataset_from_index']+j, dst['dataset_from_index']+j))
        offset += len(original)
        ep_offset += len(eps)
    assert offset == len(target)
    report['numeric_frames_exact'] = offset
    report['task_mapping_exact_episodes'] = ep_offset
    # Statistics used by the proposed 8-D training are checked against actual rows.
    stats = json.loads((merged/'meta/stats.json').read_text())
    report['stats'] = {}
    for key in ('action', 'observation.state'):
        values = np.asarray([r[key][:8] for r in target], dtype=np.float64)
        mean, std = values.mean(0), values.std(0)
        expected_mean, expected_std = np.asarray(stats[key]['mean'][:8]), np.asarray(stats[key]['std'][:8])
        report['stats'][key] = {'mean_max_error': float(abs(mean-expected_mean).max()), 'std_max_error': float(abs(std-expected_std).max())}
        np.testing.assert_allclose(mean, expected_mean, atol=2e-5, rtol=1e-4)
        np.testing.assert_allclose(std, expected_std, atol=2e-5, rtol=1e-4)
    save()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = {str(p): LeRobotDataset('local/p0', root=p, video_backend='pyav', return_uint8=True) for p in sources+[merged]}
    maxima = {}
    for n, (source, i, j) in enumerate(samples, 1):
        a, b = ds[source][i], ds[str(merged)][j]
        assert a['task'] == b['task']
        for key in ('observation.state', 'action'):
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]))
        for camera in ('front', 'left_wrist', 'right_wrist'):
            key = 'observation.images.'+camera
            err = int(np.abs(np.asarray(a[key]).astype('int16')-np.asarray(b[key]).astype('int16')).max())
            maxima[camera] = max(maxima.get(camera, 0), err)
            assert err == 0, (source, i, j, camera, err)
        if n % 50 == 0:
            report.update(decoded_frame_pairs=n, image_max_errors=maxima)
            save()
            print(f'decoded {n}/{len(samples)} frame pairs', flush=True)
    report.update(status='passed', decoded_frame_pairs=len(samples), image_max_errors=maxima,
                  limits='Five positions per episode decoded in each camera, including both boundaries. Numeric fields and task remapping checked for all frames. Original collection physics/image timing is not independently replayed.')
    report['metadata_hashes'] = {str(p/'meta/info.json'): hashlib.sha256((p/'meta/info.json').read_bytes()).hexdigest() for p in sources+[merged]}
    save()


if __name__ == '__main__':
    main()
