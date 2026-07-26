#!/usr/bin/env python3
"""Download GDM 2026 11th-edition primary cards by disposition matchup."""

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
DEFAULT_OUTPUT_DIR = Path("11eRules") / "primary_matchups"
USER_AGENT = "40kStrategemCards scraper/1.0 (+https://gdmissions.app)"

COL_RE = re.compile(
    r'<div class="fdm-colh"[^>]*>.*?<span class="fdm-nm">(.*?)</span>',
    re.DOTALL,
)
ROW_RE = re.compile(
    r'<div class="fdm-rowh"[^>]*>.*?<span class="fdm-nm">(.*?)</span>',
    re.DOTALL,
)
CELL_RE = re.compile(
    r'<div class="fdm-cell[^"]*">.*?<span class="fdm-mn">(.*?)</span>',
    re.DOTALL,
)
SITEMAP_PRIMARY_RE = re.compile(
    r"<loc>(?P<url>https://gdmissions\.app/11th/primary-missions/"
    r"(?P<deck>[^/]+)/(?P<card>[^<]+))</loc>"
)
BACK_IMAGE_RE = re.compile(
    r'\\?"back\\?":\\?"(?P<path>/assets/11th/primary-missions/[^"\\]+-back\.png)'
)


@dataclass(frozen=True)
class MatchupCard:
    you_disposition: str
    opponent_disposition: str
    card_name: str
    deck_slug: str
    card_slug: str
    source_page_url: str
    image_url: str
    local_path: str
    back_image_url: str | None
    back_local_path: str | None


@dataclass(frozen=True)
class DownloadStats:
    front_downloaded: int = 0
    front_skipped_existing: int = 0
    back_downloaded: int = 0
    back_skipped_existing: int = 0
    back_missing: int = 0

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


def clean_markup_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


def slugify(value: str) -> str:
    value = html.unescape(value).lower()
    value = value.replace("&", " and ")
    value = value.replace("'", "")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def parse_matrix(matrix_html: str) -> tuple[list[str], list[str], list[str]]:
    columns = [clean_markup_text(match.group(1)) for match in COL_RE.finditer(matrix_html)]
    rows = [clean_markup_text(match.group(1)) for match in ROW_RE.finditer(matrix_html)]
    cells = [clean_markup_text(match.group(1)) for match in CELL_RE.finditer(matrix_html)]

    columns = columns[:5]
    rows = rows[:5]
    cells = cells[: len(rows) * len(columns)]

    if len(rows) != 5 or len(columns) != 5 or len(cells) != 25:
        raise ValueError(
            "Could not parse the expected 5x5 Force Disposition Matrix "
            f"(rows={len(rows)}, columns={len(columns)}, cells={len(cells)})."
        )

    return rows, columns, cells


def parse_primary_routes(sitemap_xml: str) -> dict[tuple[str, str], str]:
    routes: dict[tuple[str, str], str] = {}
    for match in SITEMAP_PRIMARY_RE.finditer(sitemap_xml):
        routes[(match.group("deck"), match.group("card"))] = match.group("url")
    if not routes:
        raise ValueError("No 11th-edition primary mission card routes found in sitemap.xml.")
    return routes


def build_matchup_cards(
    *,
    rows: list[str],
    columns: list[str],
    cells: list[str],
    routes: dict[tuple[str, str], str],
    output_dir: Path,
    base_url: str,
    include_mirrors: bool,
) -> list[MatchupCard]:
    cards: list[MatchupCard] = []
    cell_index = 0

    for row_name in rows:
        deck_slug = slugify(row_name)
        for column_name in columns:
            card_name = cells[cell_index]
            cell_index += 1

            opponent_slug = slugify(column_name)
            if deck_slug == opponent_slug and not include_mirrors:
                continue

            card_slug = slugify(card_name)
            page_url = routes.get((deck_slug, card_slug))
            if page_url is None:
                raise ValueError(
                    "Matrix card is missing from sitemap: "
                    f"{row_name} vs {column_name} -> {card_name} "
                    f"({deck_slug}/{card_slug})"
                )

            relative_image = f"/assets/11th/primary-missions/{deck_slug}/{card_slug}.png"
            image_url = urljoin(base_url, relative_image)
            local_path = output_dir / "cards" / deck_slug / f"{card_slug}.png"

            cards.append(
                MatchupCard(
                    you_disposition=row_name,
                    opponent_disposition=column_name,
                    card_name=card_name,
                    deck_slug=deck_slug,
                    card_slug=card_slug,
                    source_page_url=page_url,
                    image_url=image_url,
                    local_path=local_path.as_posix(),
                    back_image_url=None,
                    back_local_path=None,
                )
            )

    return cards


def add_back_images(
    cards: list[MatchupCard],
    *,
    timeout: float,
    output_dir: Path,
    base_url: str,
) -> list[MatchupCard]:
    cards_with_backs: list[MatchupCard] = []

    for card in cards:
        card_html = fetch_text(card.source_page_url, timeout)
        match = BACK_IMAGE_RE.search(card_html)
        if match is None:
            cards_with_backs.append(card)
            continue

        back_path = match.group("path")
        back_url = urljoin(base_url, back_path)
        back_local_path = output_dir / "cards" / card.deck_slug / Path(back_path).name
        cards_with_backs.append(
            MatchupCard(
                you_disposition=card.you_disposition,
                opponent_disposition=card.opponent_disposition,
                card_name=card.card_name,
                deck_slug=card.deck_slug,
                card_slug=card.card_slug,
                source_page_url=card.source_page_url,
                image_url=card.image_url,
                local_path=card.local_path,
                back_image_url=back_url,
                back_local_path=back_local_path.as_posix(),
            )
        )

    return cards_with_backs


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[MatchupCard]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_pair_manifest(cards: list[MatchupCard], disposition_order: list[str]) -> dict[str, object]:
    order = {slugify(name): index for index, name in enumerate(disposition_order)}
    pairs: dict[str, object] = {}

    for card in cards:
        left, right = sorted(
            [card.deck_slug, slugify(card.opponent_disposition)],
            key=lambda slug: order[slug],
        )
        pair_key = f"{left}_vs_{right}"
        pair = pairs.setdefault(
            pair_key,
            {
                "dispositions": [
                    disposition_order[order[left]],
                    disposition_order[order[right]],
                ],
                "cards": [],
            },
        )
        pair["cards"].append(asdict(card))

    for pair in pairs.values():
        pair["cards"].sort(key=lambda card: order[slugify(card["you_disposition"])])

    return pairs


def download_png(
    url: str,
    destination: Path,
    *,
    timeout: float,
    force: bool,
    optional: bool,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        return "skipped"

    try:
        image_bytes = fetch_bytes(url, timeout=timeout)
    except HTTPError as exc:
        if optional and exc.code == 404:
            return "missing"
        raise

    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Downloaded file is not a PNG: {url}")

    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    temp_path.write_bytes(image_bytes)
    temp_path.replace(destination)
    return "downloaded"


def download_card_images(cards: list[MatchupCard], *, timeout: float, force: bool) -> DownloadStats:
    stats = DownloadStats()

    for card in cards:
        front_status = download_png(
            card.image_url,
            Path(card.local_path),
            timeout=timeout,
            force=force,
            optional=False,
        )
        if front_status == "downloaded":
            stats = stats.add("front_downloaded")
        else:
            stats = stats.add("front_skipped_existing")

        if not card.back_image_url or not card.back_local_path:
            stats = stats.add("back_missing")
            continue

        back_status = download_png(
            card.back_image_url,
            Path(card.back_local_path),
            timeout=timeout,
            force=force,
            optional=True,
        )
        if back_status == "downloaded":
            stats = stats.add("back_downloaded")
        elif back_status == "skipped":
            stats = stats.add("back_skipped_existing")
        else:
            stats = stats.add("back_missing")

    return stats


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    base_url = args.base_url.rstrip("/")
    matrix_url = urljoin(base_url, "/11th/matrix")
    sitemap_url = urljoin(base_url, "/sitemap.xml")

    matrix_html = fetch_text(matrix_url, args.timeout)
    sitemap_xml = fetch_text(sitemap_url, args.timeout)

    rows, columns, cells = parse_matrix(matrix_html)
    routes = parse_primary_routes(sitemap_xml)
    cards = build_matchup_cards(
        rows=rows,
        columns=columns,
        cells=cells,
        routes=routes,
        output_dir=output_dir,
        base_url=base_url,
        include_mirrors=args.include_mirrors,
    )

    if not cards:
        raise ValueError("No matchup cards selected.")

    if not args.skip_backs:
        cards = add_back_images(
            cards,
            timeout=args.timeout,
            output_dir=output_dir,
            base_url=base_url,
        )

    write_json(output_dir / "matchups.json", [asdict(card) for card in cards])
    write_csv(output_dir / "matchups.csv", cards)
    pairs = build_pair_manifest(cards, rows)
    write_json(output_dir / "matchup_pairs.json", pairs)

    summary = {
        "base_url": base_url,
        "matrix_url": matrix_url,
        "sitemap_url": sitemap_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_mirrors": args.include_mirrors,
        "include_backs": not args.skip_backs,
        "directed_matchup_count": len(cards),
        "unique_pair_count": len(pairs),
        "front_card_count": len(cards),
        "back_card_count": sum(1 for card in cards if card.back_image_url),
        "card_output_dir": (output_dir / "cards").as_posix(),
    }

    if args.dry_run:
        summary["front_downloaded"] = 0
        summary["front_skipped_existing"] = 0
        summary["back_downloaded"] = 0
        summary["back_skipped_existing"] = 0
        summary["back_missing"] = len(cards) - summary["back_card_count"]
        write_json(output_dir / "summary.json", summary)
        print(
            "Dry run complete: "
            f"{len(cards)} front images and {summary['back_card_count']} back images listed."
        )
        return 0

    stats = download_card_images(cards, timeout=args.timeout, force=args.force)
    summary.update(asdict(stats))
    write_json(output_dir / "summary.json", summary)

    print(
        "Complete: "
        f"{stats.front_downloaded} fronts downloaded, "
        f"{stats.front_skipped_existing} fronts skipped, "
        f"{stats.back_downloaded} backs downloaded, "
        f"{stats.back_skipped_existing} backs skipped, "
        f"{stats.back_missing} backs missing, "
        f"{len(cards)} directed matchup cards in {output_dir}."
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape 11th-edition GDM primary mission card images by Force Disposition matchup."
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
        "--include-mirrors",
        action="store_true",
        help="Include same-disposition matchups from the diagonal of the matrix.",
    )
    parser.add_argument(
        "--skip-backs",
        action="store_true",
        help="Do not discover or download optional card back images.",
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
