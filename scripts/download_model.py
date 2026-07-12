"""
Simple downloader to fetch a pretrained model file into `models/`.

Usage:
  python scripts/download_model.py --url <MODEL_URL> [--out models/deepfake_classifier.onnx]

Example:
  python scripts/download_model.py --url https://example.com/model.onnx --out models/deepfake_classifier.onnx
"""
import argparse
import os
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=False)
    p.add_argument('--hf_repo', required=False, help='Hugging Face repo ID to download')
    p.add_argument('--out', default='models/deepfake_classifier.onnx')
    args = p.parse_args()

    if not args.url and not args.hf_repo:
        raise SystemExit('Either --url or --hf_repo is required')

    if args.hf_repo:
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            raise SystemExit('Please install huggingface-hub to use --hf_repo: pip install huggingface-hub')
        os.makedirs(args.out, exist_ok=True)
        print('Downloading Hugging Face model', args.hf_repo, 'to', args.out)
        path = snapshot_download(repo_id=args.hf_repo, local_dir=args.out, local_dir_use_symlinks=False)
        print('Downloaded to', path)
        return

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    print('Downloading', args.url)
    r = requests.get(args.url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(args.out, 'wb') as f:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=8192):
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f'Progress: {downloaded/total:.2%}', end='\r')
    print('\nSaved to', args.out)


if __name__ == '__main__':
    main()
