"""Scene determinism probe: hash the post-reset scene per (seed, profile) under a given tree.

Run once per worktree, from the main repo root (outputs paths are cwd-relative):

    PYTHONPATH=<tree>/src python scripts/probe_scene_determinism.py \
        --tree <tree> --out outputs/determinism_<sha>.json

The probe wraps the task env's reset(), runs a 1-control-step episode per seed via
the tree's own CookieBatchBenchmark.run_episode, and hashes data.qpos/qvel/time plus
randomization_metadata. Same hash across trees => seed S produces the same scene in
both revisions. Import/construct/run failures are recorded per tree rather than
crashing the campaign.
"""

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path


def scene_hash(env) -> str:
    h = hashlib.sha256()
    h.update(b"qpos:")
    h.update(env.data.qpos.astype("<f8").tobytes())
    h.update(b"qvel:")
    h.update(env.data.qvel.astype("<f8").tobytes())
    h.update(str(env.data.time).encode())
    meta = getattr(env, "randomization_metadata", None)
    if meta is not None:
        h.update(b"meta:" + json.dumps(meta, sort_keys=True, default=str).encode())
    return h.hexdigest()


def build_benchmark(tree: Path, profile: str, notes: list):
    from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark

    params = inspect.signature(CookieBatchBenchmark.__init__).parameters
    kwargs = {"max_steps": 1}
    settings = None
    try:
        from a3_dual_arm_sim.sim.randomization import load_randomization_config

        cfg = tree / "configs" / "randomization.yaml"
        if cfg.is_file():
            settings = load_randomization_config(cfg)
            settings["profile"] = profile
            settings["source_column"] = "1"
    except Exception as exc:  # module moved (sim/ vs top level) across revisions
        notes.append(f"settings-import/{profile}: {exc!r}")
    if "profile" in params:
        kwargs["profile"] = profile
    if "source_column" in params:
        kwargs["source_column"] = "1"
    if settings is not None and "randomization_settings" in params:
        kwargs["randomization_settings"] = settings
    return CookieBatchBenchmark(tree / "configs" / "cookie_batch.yaml", **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=13000)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--profiles", nargs="+", default=["basic", "medium", "advanced"])
    args = ap.parse_args()
    tree = args.tree.resolve()

    import a3_dual_arm_sim

    pkg = Path(a3_dual_arm_sim.__file__).resolve()
    result = {
        "tree": str(tree),
        "package": str(pkg),
        "in_tree": pkg.is_relative_to(tree),
        "argv_python": sys.version.split()[0],
        "revision": subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "rows": {},
        "notes": [],
    }
    if not result["in_tree"]:
        result["notes"].append("PACKAGE_RESOLVED_OUTSIDE_TREE")
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
        return 2

    try:
        from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
    except Exception as exc:
        result["fatal"] = f"import-env: {exc!r}"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
        return 1

    captured = {}
    original_reset = A3CookieTransferEnv.reset

    def wrapped_reset(self, **kwargs):
        observation, info = original_reset(self, **kwargs)
        captured["hash"] = scene_hash(self)
        return observation, info

    A3CookieTransferEnv.reset = wrapped_reset
    try:
        for profile in args.profiles:
            try:
                benchmark = build_benchmark(tree, profile, result["notes"])
            except Exception as exc:
                result["notes"].append(f"construct/{profile}: {exc!r}")
                continue
            consecutive_failures = 0
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                captured.clear()
                try:
                    benchmark.run_episode("same_column", seed)
                except Exception as exc:
                    result["notes"].append(f"run/{profile}/{seed}: {exc!r}")
                    consecutive_failures += 1
                    if consecutive_failures >= 8:
                        break  # tree lacks the expected API shape; stop this profile
                    continue
                if "hash" not in captured:
                    result["notes"].append(f"no-reset/{profile}/{seed}")
                    consecutive_failures += 1
                    if consecutive_failures >= 8:
                        break
                    continue
                consecutive_failures = 0
                result["rows"][f"{profile}/{seed}"] = captured["hash"]
    finally:
        A3CookieTransferEnv.reset = original_reset

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"{result['revision'][:9]} rows={len(result['rows'])} "
        f"notes={len(result['notes'])} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
