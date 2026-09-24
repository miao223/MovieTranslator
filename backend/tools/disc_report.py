"""What would the 原盘 page do with this disc? Print it, title by title.

    cd backend
    .venv/bin/python -m tools.disc_report /path/to/Some.Film
    .venv/bin/python -m tools.disc_report /path/to/Show.Vol.1 --series yes --episode-start 5
    .venv/bin/python -m tools.disc_report --batch /path/to/discs     # every disc in a folder

Works on a metadata-only copy of a disc too (BDMV without STREAM/, or just
the IFOs) — which is how real discs are collected for calibrating the
thresholds in app/services/disc/analyze.py without copying the films.
"""

from __future__ import annotations

import argparse
import sys

from app.models.schemas import AppSettings, DiscSettings
from app.services.disc import report as disc_report
from app.services.disc.binary import DiscError
from app.services.disc.fs import find_discs
from app.services.disc.plan import Answer, output_choice, plan, report_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--batch", action="store_true",
                    help="path 是文件夹：分析里面的每一张盘（原盘页的批量模式）")
    ap.add_argument("--series", choices=("yes", "no"))
    ap.add_argument("--episode-start", type=int, default=None, help="默认自动（多卷合集接着前一卷）")
    ap.add_argument("--name", default="")
    ap.add_argument("--no-extras", action="store_true", help="花絮默认不勾")
    ap.add_argument("--min-seconds", type=int, default=60)
    ap.add_argument("--output-mode", choices=("beside", "inside", "custom"), default="beside")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args(argv)
    series = None if args.series is None else args.series == "yes"
    settings = AppSettings(disc=DiscSettings(export_extras=not args.no_extras,
                                             min_title_seconds=args.min_seconds))
    mode, custom = output_choice(args.output_mode, args.output_dir, settings)
    if args.batch:
        try:
            paths = [str(p) for p in find_discs(args.path)]
        except DiscError as exc:
            print(f"✕ {exc}", file=sys.stderr)
            return 2
    else:
        paths = [args.path]
    answer = Answer(series, args.name, args.episode_start)
    status = 0
    for n, item in enumerate(plan([(p, answer) for p in paths], settings)):
        if n:
            print()
        if item.disc is None:
            print(f"✕ {item.path}：{item.error}", file=sys.stderr)
            status = 2
            continue
        report = report_of(item, settings, mode, custom)
        for line in disc_report.lines(report):
            print(line)
        if item.analysis.series_choice:
            other = "no" if item.analysis.mode == "series" else "yes"
            print(f"\n（这张盘可以在整片/分集之间切换：加 --series {other} 看另一种）")
    return status


if __name__ == "__main__":
    sys.exit(main())
