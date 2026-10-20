"""Compare SDPA/eager on real checkpoint batches with identical RNG and weights."""
import argparse
import copy
import json
import types
from pathlib import Path

import torch
from torch.utils.data import default_collate

from a3_dual_arm_sim.learning.checkpoint_bundle import load_processors
from a3_dual_arm_sim.learning.smolvla_attention import sdpa_attention_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    config.device = 'cuda'
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=config).cuda()
    pre, post = load_processors(config, args.checkpoint, None, 'cuda')
    ds = LeRobotDataset('local/p0', root=args.dataset, video_backend='pyav', delta_timestamps={'action': [i/20 for i in range(config.chunk_size)]})
    original = SmolVLMWithExpertModel.get_attention_interface
    def set_mode(mode):
        if mode == 'eager':
            SmolVLMWithExpertModel.get_attention_interface = original
        else:
            SmolVLMWithExpertModel.get_attention_interface = lambda self: types.MethodType(sdpa_attention_forward, self)
    report = {'status': 'running', 'checkpoint': str(args.checkpoint), 'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'cases': [], 'tolerances': {'loss_relative': 0.02, 'gradient_relative_l2': 0.05, 'inference_action_absolute': 0.01}}
    def save():
        args.report.write_text(json.dumps(report, indent=2))
    save()
    try:
        for ep in (0, 100):
            index = int(ds.meta.episodes['dataset_from_index'][ep])+20
            batch = pre(default_collate([ds[index]]))
            # The loaded model has mixed parameter dtypes even without autocast.
            for amp in (False, True):
                values = {}
                for mode in ('eager', 'sdpa'):
                    set_mode(mode)
                    policy.train()
                    policy.zero_grad(set_to_none=True)
                    torch.manual_seed(20261016)
                    torch.cuda.manual_seed_all(20261016)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                        loss, _ = policy(copy.deepcopy(batch))
                    loss.backward()
                    grads = {n: p.grad.detach().float().cpu().clone() for n,p in policy.named_parameters() if p.requires_grad and p.grad is not None}
                    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads.values())
                    values[mode] = (float(loss.detach()), grads)
                a, b = values['eager'], values['sdpa']
                assert a[1].keys() == b[1].keys()
                diff2 = sum(float(((a[1][n]-b[1][n])**2).sum()) for n in a[1])
                norm2 = sum(float((a[1][n]**2).sum()) for n in a[1])
                item = {'episode': ep, 'frame': 20, 'amp_bfloat16': amp, 'eager_loss': a[0], 'sdpa_loss': b[0], 'loss_relative': abs(a[0]-b[0])/max(abs(a[0]),1e-12), 'gradient_relative_l2': (diff2/max(norm2,1e-24))**0.5, 'gradient_tensors': len(a[1])}
                item['passed'] = item['loss_relative'] < 0.02 and item['gradient_relative_l2'] < 0.05
                report['cases'].append(item)
                print(json.dumps(item), flush=True)
                save()
                del values, a, b, grads
                policy.zero_grad(set_to_none=True)
            outputs = {}
            for mode in ('eager', 'sdpa'):
                set_mode(mode)
                policy.eval()
                policy.reset()
                torch.manual_seed(20261016)
                torch.cuda.manual_seed_all(20261016)
                with torch.inference_mode():
                    outputs[mode] = post(policy.predict_action_chunk(copy.deepcopy(batch))).detach().float().cpu()
            err = float((outputs['sdpa']-outputs['eager']).abs().max())
            report['cases'].append({'episode': ep, 'inference_action_max_abs': err, 'passed': err < 0.01})
            save()
        report['status'] = 'passed' if all(c['passed'] for c in report['cases']) else 'failed'
        report['limits'] = 'Two real observations, fixed RNG, autocast-disabled (native mixed parameter dtypes) and bf16-autocast loss/gradient; 25-step flow inference. Numerical agreement does not establish learned policy performance.'
        save()
    finally:
        SmolVLMWithExpertModel.get_attention_interface = original


if __name__ == '__main__':
    main()
