"""Bounded Modal launcher for microK3. Run: modal run modal_train.py --steps 500"""

from pathlib import Path

import modal

MAX_CORPUS_BYTES = 8 << 20

app = modal.App("microk3-teaching-run")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.2,<3")
    .add_local_file("microk3.py", "/root/microk3.py")
)


@app.function(image=image, gpu="L40S", timeout=30 * 60)
def train(steps: int, batch_size: int, block_size: int, optimizer: str, data: bytes | None):
    import subprocess

    if not 1 <= steps <= 10_000:
        raise ValueError("steps must be between 1 and 10,000 (cost safety cap)")
    command = ["python", "/root/microk3.py", "--device", "cuda", "--steps", str(steps)]
    command += ["--batch-size", str(batch_size), "--block-size", str(block_size)]
    command += ["--optimizer", optimizer]
    if data:
        Path("/root/corpus.bin").write_bytes(data)
        command += ["--data", "/root/corpus.bin"]
    subprocess.run(command, check=True)


@app.local_entrypoint()
def main(
    steps: int = 500,
    batch_size: int = 8,
    block_size: int = 128,
    optimizer: str = "muon",
    data: str = "",
):
    """Ship a small corpus with the run; anything larger belongs in a Modal Volume."""
    payload = Path(data).read_bytes() if data else None
    if payload is not None and len(payload) > MAX_CORPUS_BYTES:
        raise ValueError(f"corpus must be at most {MAX_CORPUS_BYTES:,} bytes, got {len(payload):,}")
    train.remote(steps, batch_size, block_size, optimizer, payload)
