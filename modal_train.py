"""Bounded Modal launcher for microK3. Run: modal run modal_train.py --steps 500"""

import modal

app = modal.App("microk3-teaching-run")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.2").add_local_file(
    "microk3.py", "/root/microk3.py"
)


@app.function(image=image, gpu="L40S", timeout=30 * 60)
def train(steps: int):
    import subprocess

    if not 1 <= steps <= 10_000:
        raise ValueError("steps must be between 1 and 10,000 (cost safety cap)")
    subprocess.run(["python", "/root/microk3.py", "--steps", str(steps), "--device", "cuda"], check=True)


@app.local_entrypoint()
def main(steps: int = 500):
    train.remote(steps)
