"""List Kimi K3 repository metadata without downloading any model shard."""

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

REPO = "moonshotai/Kimi-K3"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def main() -> None:
    files = HfApi().list_repo_files(REPO)
    print(f"{REPO}: {len(files)} remote files (metadata only)\n")
    total = 0
    for name in files:
        if name.endswith(WEIGHT_SUFFIXES):
            metadata = get_hf_file_metadata(hf_hub_url(REPO, name))  # HEAD request, no body
            size = metadata.size or 0
            total += size
            print(f"WEIGHT  {size / 2**30:8.2f} GiB  {name}")
        else:
            print(f"CONTROL {'—':>8}      {name}")
    print(f"\nRemote weight total from HEAD metadata: {total / 2**40:.3f} TiB")
    print("Downloaded: 0 bytes. This script has no download call.")


if __name__ == "__main__":
    main()
