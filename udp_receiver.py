"""Small UDP receiver used to validate the tracking server packet stream."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("runs/udp_test/received.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    received: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind((args.host, args.port))
        server.settimeout(args.timeout)
        print(f"Listening on {args.host}:{args.port}; waiting for {args.count} packets")
        while len(received) < args.count:
            try:
                payload, address = server.recvfrom(4096)
            except socket.timeout:
                break
            text = payload.decode("utf-8", errors="replace").rstrip("\n")
            received.append(text)
            if len(received) <= 3:
                print(f"{address}: {text}")
    args.output.write_text("\n".join(received) + ("\n" if received else ""), encoding="utf-8")
    print(f"received={len(received)} expected={args.count}")
    print(f"Saved packets: {args.output}")
    if len(received) != args.count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
