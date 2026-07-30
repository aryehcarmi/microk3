"""Bounded Modal launcher for microK3. Run: modal run modal_train.py --steps 500"""

from pathlib import Path

import modal

MAX_CORPUS_BYTES = 8 << 20

app = modal.App("microk3-teaching-run")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4,<3")
    .add_local_file("microk3.py", "/root/microk3.py")
)


@app.function(image=image, gpu="L40S", timeout=30 * 60)
def train(
    steps: int,
    batch_size: int,
    block_size: int,
    optimizer: str,
    generate: int,
    data: bytes | None,
):
    import subprocess

    if not 1 <= steps <= 10_000:
        raise ValueError("steps must be between 1 and 10,000 (cost safety cap)")
    if not 0 <= generate <= 1_000:
        raise ValueError("generate must be between 0 and 1,000")
    if data is not None and len(data) > MAX_CORPUS_BYTES:
        raise ValueError(f"corpus must be at most {MAX_CORPUS_BYTES:,} bytes, got {len(data):,}")
    command = ["python", "/root/microk3.py", "--device", "cuda", "--steps", str(steps)]
    command += ["--batch-size", str(batch_size), "--block-size", str(block_size)]
    command += ["--optimizer", optimizer, "--generate", str(generate)]
    if data is not None:
        Path("/root/corpus.bin").write_bytes(data)
        command += ["--data", "/root/corpus.bin"]
    subprocess.run(command, check=True)


@app.local_entrypoint()
def main(
    steps: int = 500,
    batch_size: int = 8,
    block_size: int = 128,
    optimizer: str = "muon",
    generate: int = 0,
    data: str = "",
):
    """Ship a small corpus with the run; anything larger belongs in a Modal Volume."""
    path = Path(data) if data else None
    if path is not None and path.stat().st_size > MAX_CORPUS_BYTES:
        raise ValueError(f"corpus must be at most {MAX_CORPUS_BYTES:,} bytes")
    payload = path.read_bytes() if path is not None else None
    train.remote(steps, batch_size, block_size, optimizer, generate, payload)
