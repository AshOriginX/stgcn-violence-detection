"""
Compare phase4 vs domain-adapted checkpoint on CCTV demo videos.
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.inference import (
    TemporalDecisionConfig,
    initialize_pipeline,
    load_checkpoint,
    load_config,
    run_inference_on_video,
)
import torch


def run_comparison(
    checkpoint_path: Path,
    cctv_dir: Path,
    pipeline_config: Path,
    device: str,
    decision_config: TemporalDecisionConfig,
    save_video: bool = False,
) -> list:
    checkpoint, model = load_checkpoint(checkpoint_path)
    pipeline_cfg = load_config(pipeline_config)
    extractor, exporter = initialize_pipeline(pipeline_cfg)
    dev = torch.device(device)

    results = []
    for video_path in sorted(cctv_dir.glob("cctv*.mp4")):
        result = run_inference_on_video(
            video_path=video_path,
            model=model,
            extractor=extractor,
            exporter=exporter,
            device=dev,
            save_video=save_video,
            decision_config=decision_config,
        )
        result["checkpoint"] = str(checkpoint_path)
        results.append(result)

    extractor.close()
    exporter.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints on CCTV videos")
    parser.add_argument(
        "--phase4-checkpoint",
        type=Path,
        default=Path("outputs/experiment_phase4_reduce_lr_on_plateau/best.pt"),
    )
    parser.add_argument(
        "--adapted-checkpoint",
        type=Path,
        default=Path("outputs/experiment_college_domain_adaptation/best.pt"),
    )
    parser.add_argument(
        "--cctv-dir",
        type=Path,
        default=Path("sample_videos_for_testing"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cctv_comparison.json"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    decision_config = TemporalDecisionConfig()
    pipeline_config = Path("configs/pipeline.yaml")

    all_results = {}

    print("=" * 60)
    print("PHASE-4 CHECKPOINT")
    print("=" * 60)
    phase4_results = run_comparison(
        args.phase4_checkpoint, args.cctv_dir, pipeline_config,
        args.device, decision_config, args.save_video,
    )
    all_results["phase4"] = phase4_results

    for r in phase4_results:
        if r.get("success"):
            print(
                f"  {r['video_id']}: avg={r['avg_fight_probability']:.3f} "
                f"max={r['max_fight_probability']:.3f} "
                f"confirmed={r['confirmed_fight_windows']} "
                f"state={r['final_state']}"
            )

    if args.adapted_checkpoint.exists():
        print("\n" + "=" * 60)
        print("DOMAIN-ADAPTED CHECKPOINT")
        print("=" * 60)
        adapted_results = run_comparison(
            args.adapted_checkpoint, args.cctv_dir, pipeline_config,
            args.device, decision_config, args.save_video,
        )
        all_results["adapted"] = adapted_results
        for r in adapted_results:
            if r.get("success"):
                print(
                    f"  {r['video_id']}: avg={r['avg_fight_probability']:.3f} "
                    f"max={r['max_fight_probability']:.3f} "
                    f"confirmed={r['confirmed_fight_windows']} "
                    f"state={r['final_state']}"
                )
    else:
        print(f"\nAdapted checkpoint not found: {args.adapted_checkpoint}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
