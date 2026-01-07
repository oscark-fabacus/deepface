#!/usr/bin/env python3
"""Generate ArcFace embeddings for a directory of cropped face images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepface import DeepFace


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ArcFace embeddings for face crops in a directory."
    )
    parser.add_argument("faces_dir", help="Directory containing face crop images.")
    parser.add_argument(
        "--output",
        default="arcface_embeddings.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--enforce-detection",
        action="store_true",
        default=False,
        help="Raise an error if no face is detected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    faces_dir = Path(args.faces_dir)
    output_path = Path(args.output)

    results = []
    for image_path in sorted(faces_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        representation = DeepFace.represent(
            img_path=str(image_path),
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=args.enforce_detection,
        )
        if not representation:
            continue
        results.append(
            {
                "image_path": str(image_path),
                "embedding": representation[0]["embedding"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
