"""List Kimi K3 repository metadata without downloading any model shard."""

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

REPO = "moonshotai/Kimi-K3"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def main() -> None:
    files = HfApi(token=False).list_repo_files(repo_id=REPO, repo_type="model")
    print(f"{REPO}: {len(files)} remote files (metadata only)\n")
    total = 0
    weight_count = 0
    unknown_sizes = []
    for name in files:
        if name.endswith(WEIGHT_SUFFIXES):
            metadata = get_hf_file_metadata(hf_hub_url(REPO, name), token=False)  # HEAD request, no body
            weight_count += 1
            if metadata.size is None:
                unknown_sizes.append(name)
                print(f"WEIGHT  {'unknown':>8}      {name}")
                continue
            size = metadata.size
            total += size
            print(f"WEIGHT  {size / 2**30:8.2f} GiB  {name}")
        else:
            print(f"CONTROL {'-':>8}      {name}")
    print(f"\nRecognized weight files: {weight_count}")
    print(f"Known remote weight total from HEAD metadata: {total / 2**40:.3f} TiB")
    if unknown_sizes:
        print(f"Files with unknown sizes: {len(unknown_sizes)}")
    print("Downloaded file bodies: 0 bytes.")


if __name__ == "__main__":
    main()
