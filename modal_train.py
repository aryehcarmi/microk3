"""Bounded Modal launcher for microK3. Run: modal run modal_train.py --steps 500"""

from pathlib import Path

import modal

MAX_CORPUS_BYTES = 8 << 20

app = modal.App("microk3-teaching-run")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4.1,<3")
    .add_local_file("microk3.py", "/root/microk3.py")
)


@app.function(image=image, gpu="L40S", timeout=30 * 60)
def train(
    steps: int,
    batch_size: int,
    block_size: int,
    optimizer: str,
    generate: int = 0,
    data: bytes | None = None,
    vision: bool = False,
    quantize: bool = False,
):
    import subprocess

    if not 1 <= steps <= 10_000:
        raise ValueError("steps must be between 1 and 10,000 (cost safety cap)")
    if not 0 <= generate <= 1_000:
        raise ValueError("generate must be between 0 and 1,000")
    if data is not None and len(data) > MAX_CORPUS_BYTES:
        raise ValueError(f"corpus must be at most {MAX_CORPUS_BYTES:,} bytes, got {len(data):,}")
    if vision and data is not None:
        raise ValueError("--data cannot be combined with --vision; the shapes task draws its own pictures")
    command = ["python", "/root/microk3.py", "--device", "cuda", "--steps", str(steps)]
    command += ["--batch-size", str(batch_size), "--block-size", str(block_size)]
    command += ["--optimizer", optimizer, "--generate", str(generate)]
    command += ["--vision"] if vision else []
    command += ["--quantize"] if quantize else []
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
    vision: bool = False,
    quantize: bool = False,
):
    """Ship a small corpus with the run; anything larger belongs in a Modal Volume."""
    if data and vision:
        raise ValueError("--data cannot be combined with --vision; the shapes task draws its own pictures")
    path = Path(data) if data else None
    if path is None:
        payload = None
    else:
        if not path.is_file():
            raise ValueError("corpus path must be a regular file")
        with path.open("rb") as corpus:
            payload = corpus.read(MAX_CORPUS_BYTES + 1)
        if len(payload) > MAX_CORPUS_BYTES:
            raise ValueError(f"corpus must be at most {MAX_CORPUS_BYTES:,} bytes")
    train.remote(
        steps=steps,
        batch_size=batch_size,
        block_size=block_size,
        optimizer=optimizer,
        generate=generate,
        data=payload,
        vision=vision,
        quantize=quantize,
    )
