#!/usr/bin/env python3
"""Download GDM 2026 11th-edition secondary mission cards."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://gdmissions.app"
DEFAULT_OUTPUT_DIR = Path("11eRules") / "secondary_missions"
USER_AGENT = "40kStrategemCards scraper/1.0 (+https://gdmissions.app)"
SECONDARY_ROUTE_RE = re.compile(
    r"<loc>(https://gdmissions\.app/11th/secondary-missions/(?P<slug>[^<]+)-defender)</loc>"
)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class SecondaryCard:
    card_name: str
    card_slug: str
    side: str
    source_page_url: str
    image_url: str
    local_path: str


@dataclass(frozen=True)
class DownloadStats:
    downloaded: int = 0
    skipped_existing: int = 0

    def add(self, field: str) -> "DownloadStats":
        values = asdict(self)
        values[field] += 1
        return DownloadStats(**values)


def fetch_text(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_secondary_routes(sitemap_xml: str) -> list[tuple[str, str]]:
    routes = [
        (match.group("slug"), match.group(1))
        for match in SECONDARY_ROUTE_RE.finditer(sitemap_xml)
    ]
    if not routes:
        raise ValueError("No 11th-edition defender secondary mission routes found in sitemap.xml.")
    return sorted(routes, key=lambda route: route[0])


def parse_card_name(card_html: str, fallback_slug: str) -> str:
    match = TITLE_RE.search(card_html)
    if not match:
        return fallback_slug.replace("-", " ").title()

    title = html.unescape(match.group(1)).strip()
    title = re.sub(r"\s+-\s+Defender Secondary\s+\|\s+GDM 2026$", "", title)
    return title


def build_secondary_cards(
    *,
    routes: list[tuple[str, str]],
    base_url: str,
    output_dir: Path,
    sides: list[str],
    timeout: float,
) -> list[SecondaryCard]:
    cards: list[SecondaryCard] = []

    for card_slug, source_page_url in routes:
        card_html = fetch_text(source_page_url, timeout)
        card_name = parse_card_name(card_html, card_slug)

        for side in sides:
            relative_image = f"/assets/11th/secondary-missions/{side}/{card_slug}.png"
            image_url = urljoin(base_url, relative_image)
            local_path = output_dir / "cards" / side / f"{card_slug}.png"
            cards.append(
                SecondaryCard(
                    card_name=card_name,
                    card_slug=card_slug,
                    side=side,
                    source_page_url=source_page_url,
                    image_url=image_url,
                    local_path=local_path.as_posix(),
                )
            )

    return cards


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[SecondaryCard]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_grouped_manifest(path: Path, cards: list[SecondaryCard]) -> None:
    grouped: dict[str, dict[str, object]] = {}
    for card in cards:
        entry = grouped.setdefault(
            card.card_slug,
            {
                "card_name": card.card_name,
                "card_slug": card.card_slug,
                "sides": {},
            },
        )
        entry["sides"][card.side] = asdict(card)
    write_json(path, grouped)


def download_png(url: str, destination: Path, *, timeout: float, force: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return "skipped"

    image_bytes = fetch_bytes(url, timeout=timeout)
    if not image_bytes.startswith(PNG_SIGNATURE):
        raise ValueError(f"Downloaded file is not a PNG: {url}")

    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    temp_path.write_bytes(image_bytes)
    temp_path.replace(destination)
    return "downloaded"


def download_cards(cards: list[SecondaryCard], *, timeout: float, force: bool) -> DownloadStats:
    stats = DownloadStats()

    for card in cards:
        status = download_png(
            card.image_url,
            Path(card.local_path),
            timeout=timeout,
            force=force,
        )
        if status == "downloaded":
            stats = stats.add("downloaded")
        else:
            stats = stats.add("skipped_existing")

    return stats


def parse_sides(value: str) -> list[str]:
    sides = [part.strip().lower() for part in value.split(",") if part.strip()]
    invalid = sorted(set(sides) - {"attacker", "defender"})
    if invalid:
        raise argparse.ArgumentTypeError(
            "Sides must be attacker, defender, or attacker,defender. "
            f"Invalid: {', '.join(invalid)}"
        )
    return sides


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    base_url = args.base_url.rstrip("/")
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    sitemap_xml = fetch_text(sitemap_url, args.timeout)
    routes = parse_secondary_routes(sitemap_xml)
    cards = build_secondary_cards(
        routes=routes,
        base_url=base_url,
        output_dir=output_dir,
        sides=args.sides,
        timeout=args.timeout,
    )

    if not cards:
        raise ValueError("No secondary cards selected.")

    write_json(output_dir / "secondaries.json", [asdict(card) for card in cards])
    write_csv(output_dir / "secondaries.csv", cards)
    write_grouped_manifest(output_dir / "secondary_cards.json", cards)

    summary = {
        "base_url": base_url,
        "sitemap_url": sitemap_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sides": args.sides,
        "unique_card_count": len(routes),
        "image_count": len(cards),
        "card_output_dir": (output_dir / "cards").as_posix(),
    }

    if args.dry_run:
        summary["downloaded"] = 0
        summary["skipped_existing"] = 0
        write_json(output_dir / "summary.json", summary)
        print(f"Dry run complete: {len(cards)} secondary images listed.")
        return 0

    stats = download_cards(cards, timeout=args.timeout, force=args.force)
    summary.update(asdict(stats))
    write_json(output_dir / "summary.json", summary)
    print(
        "Complete: "
        f"{stats.downloaded} downloaded, {stats.skipped_existing} skipped, "
        f"{len(cards)} secondary images in {output_dir}."
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape 11th-edition GDM secondary mission card images."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Destination directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"Source site base URL. Default: {BASE_URL}",
    )
    parser.add_argument(
        "--sides",
        type=parse_sides,
        default=["attacker", "defender"],
        help="Comma-separated sides to download. Default: attacker,defender",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload images even when the target file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests without downloading PNG files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds. Default: 30",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv or sys.argv[1:]))
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
