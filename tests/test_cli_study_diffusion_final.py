import subprocess
import sys
import yaml
from pathlib import Path



def test_study_v1_diffusion_final_eval(tmp_path):
    """
    Minimal & strict: ensure
      - schema: study/v1 loads
      - diffusion.enabled: true builds the diffusion path
      - eval.final.enabled: true triggers final evaluation
      - final sampler settings are applied (ddpm, nfe=50)
      - run completes end-to-end
    """

    cfg = {
        "schema": "study/v1",
        "study_name": "study_v1_diffusion_final_test",
        "metric": "val/fid",
        "seed": 0,

        "data": {
            "dataset": "cifar10",
            "subset": 8,          # tiny for test
            "batch_size": 4
        },

        "model": {"name": "unet_cifar32"},

        "ema": {"enabled": True, "decay": 0.999},

        "diffusion": {
            "enabled": True,
            "beta_schedule": "linear",
        },

        "optim": {"optimizer": "adam", "lr": 1e-4},

        "train": {
            "total_steps": 5,        # tiny smoke
            "amp": False,
        },

        "eval": {
            "final": {
                "enabled": True,
                "at_end": True,
                "sampler": "ddpm",
                "nfe": 50,
                "n_samples": 2,      # keep tiny
                "fid_stats": "external/ablation-harness/stats/cifar10_inception_train.npz"
            }
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Run CLI: adjust this to your actual entrypoint
    cmd = [
        sys.executable,
        "-m",
        "ablation_harness.cli", "run",    # or your actual train entrypoint
        "--config",
        str(cfg_path),
        "--out_dir",
        tmp_path,
    ]

    result = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )


    # If failure, show stdout/stderr in pytest output
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    assert result.returncode == 0, "Training run failed."

    # Check logs for diffusion + final eval usage
    out = result.stdout + result.stderr

    assert "final evaluation" in out.lower() or "eval.final" in out.lower(), \
        "Final evaluation did not trigger."

    assert "sampler=ddpm" in out.lower(), "Final sampler mismatch."
    assert "nfe=50" in out.lower(), "Final NFE=50 not applied."
