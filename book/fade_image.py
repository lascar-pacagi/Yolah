#!/usr/bin/env python3
"""Generate a faded version of an image by setting its alpha channel."""

import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Generate a faded image")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--alpha", type=int, default=40,
                        help="Alpha value 0-255 (default: 40)")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGBA")
    data = img.getdata()
    img.putdata([(r, g, b, args.alpha) for r, g, b, a in data])
    img.save(args.output)
    print(f"{args.input} -> {args.output} (alpha={args.alpha}/255)")

if __name__ == "__main__":
    main()
