import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from subprocess import run

    reset = True

    if reset:
        run("rm -rf out/ && mkdir -p out/", shell=True, check=True)
    return (run,)


@app.cell
def _(run):
    time_command = "/usr/bin/time -l"
    ffmpeg_command = "ffmpeg"
    ffprobe_command = "ffprobe"
    video_filter = "scale=720:1280:flags=lanczos,format=yuv420p"

    cases = [
        # ref: https://code.videolan.org/videolan/x264/-/blob/master/x264.h#L704
        # 小さい方が早いが品質が落ちる
        ("libx264", "compat", "1"),
        ("libx264", "compat", "3"),
        ("libx264", "compat", "5"),  # default
        ("libx264", "compat", "7"),
        ("libx264", "compat", "9"),
        # ref: https://x265.readthedocs.io/en/stable/presets.html
        # 小さい方が早いが品質が落ちる
        ("libx265", "compat", "1"),
        ("libx265", "compat", "3"),
        ("libx265", "compat", "5"),  # default
        ("libx265", "compat", "7"),
        ("libx265", "compat", "9"),
        # ref: https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Docs/CommonQuestions.md#what-presets-do
        # 小さい方が高品質/高圧縮だがエンコードに時間がかかる
        # Generally speaking, presets 1-3 represent extremely high efficiency, for use when encode time is not important and quality/size of the resulting video file is critical.
        # Presets 4-6 are commonly used by home enthusiasts as they represent a balance of efficiency and reasonable compute time.
        # Presets between 7 and 13 are used for fast and real-time encoding. One should use the lowest preset that is tolerable.
        ("libsvtav1", "compat", "4"),
        ("libsvtav1", "compat", "6"),
        ("libsvtav1", "compat", "8"),  # default
        ("libsvtav1", "compat", "10"),
        ("libsvtav1", "compat", "12"),
    ]
    for codec, input_type, preset in cases:
        print(f"convert: {codec=} {input_type=} {preset=}")
        out_prefix = f"out/{codec}_{input_type}_preset{preset}"

        run(
            f"""
        {time_command} {ffmpeg_command} -y -i movies/{input_type}.mov \
            -vf "{video_filter}" \
            -c:v {codec} -preset {preset} \
            -an \
            {out_prefix}.mp4 > {out_prefix}_ffmpeg.log 2>&1""",
            shell=True,
            check=True,
        )

        run(
            f"""
        {ffprobe_command} {out_prefix}.mp4 > {out_prefix}_ffprobe.txt 2>&1
        """,
            shell=True,
            check=True,
        )

        run(
            f"""
        {ffmpeg_command} -i {out_prefix}.mp4 -i movies/{input_type}.mov \
            -lavfi "[0:v]{video_filter}[dist];[1:v]{video_filter}[ref];[dist][ref]libvmaf=log_fmt=json:log_path={out_prefix}_vmaf.json" \
            -f null - > {out_prefix}_vmaf.txt 2>&1
        """,
            shell=True,
            check=True,
        )
    return


@app.cell
def _():
    import json
    import re
    import subprocess
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Optional

    import pandas as pd

    # ========== 設定 ==========
    OUT_DIR = Path("out")
    MOVIES_DIR = Path("movies")

    INPUT_SUFFIX = ".mov"

    # ========== パース系ユーティリティ ==========
    _TIME_RE = re.compile(
        r"^\s*(?P<real>[\d.]+)\s+real\s+(?P<user>[\d.]+)\s+user\s+(?P<sys>[\d.]+)\s+sys\s*$"
    )

    _PREFIX_RE = re.compile(
        r"^(?P<codec>[^_]+)_(?P<input_type>[^_]+)_preset(?P<preset>\d+)$"
    )

    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    def parse_time_real_user_sys(
        ffmpeg_log_path: Path,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        text = _read_text(ffmpeg_log_path)
        for line in reversed(text.splitlines()[-200:]):
            m = _TIME_RE.match(line)
            if m:
                return float(m["real"]), float(m["user"]), float(m["sys"])
        return None, None, None

    def extract_vmaf_mean(vmaf_json_path: Path) -> Optional[float]:
        try:
            data = json.loads(vmaf_json_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        frames = data.get("frames")
        if not isinstance(frames, list) or not frames:
            return None

        preferred_keys = ["vmaf", "vmaf_v0.6.1", "vmaf_v0.6.1neg", "vmaf_v0.6.1_nneg"]

        values: list[float] = []
        for fr in frames:
            metrics = fr.get("metrics", {}) if isinstance(fr, dict) else {}
            if not isinstance(metrics, dict):
                continue

            key = None
            for k in preferred_keys:
                if k in metrics:
                    key = k
                    break
            if key is None:
                for k in metrics.keys():
                    if isinstance(k, str) and "vmaf" in k.lower():
                        key = k
                        break
            if key is None:
                continue

            v = metrics.get(key)
            if isinstance(v, (int, float)):
                values.append(float(v))

        if not values:
            return None
        return sum(values) / len(values)

    def file_size_bytes(path: Path) -> Optional[int]:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return None

    def get_duration_seconds_ffprobe(path: Path) -> Optional[float]:
        """
        Return media duration in seconds using ffprobe.
        Requires ffprobe in PATH (typically bundled with ffmpeg).
        """
        if not path.exists():
            return None
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, text=True
            ).strip()
            if not out:
                return None
            v = float(out)
            return v if v > 0 else None
        except Exception:
            return None

    # ========== 収集本体 ==========
    def collect_codec_benchmark_rows(
        out_dir: Path = OUT_DIR, movies_dir: Path = MOVIES_DIR
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        # Cache duration per src_path to avoid calling ffprobe repeatedly
        src_duration_cache: dict[str, Optional[float]] = {}

        for ffmpeg_log in sorted(out_dir.glob("*_ffmpeg.log")):
            stem = ffmpeg_log.name.removesuffix("_ffmpeg.log")
            m = _PREFIX_RE.match(stem)
            if not m:
                continue

            codec = m["codec"]
            input_type = m["input_type"]
            preset = int(m["preset"])

            prefix = out_dir / stem
            mp4_path = Path(str(prefix) + ".mp4")
            vmaf_json_path = Path(str(prefix) + "_vmaf.json")

            # 元動画
            src_path = movies_dir / f"{input_type}{INPUT_SUFFIX}"

            src_size = file_size_bytes(src_path)
            out_size = file_size_bytes(mp4_path)
            size_ratio = (out_size / src_size) if (src_size and out_size) else None

            # duration
            src_key = str(src_path)
            if src_key not in src_duration_cache:
                src_duration_cache[src_key] = get_duration_seconds_ffprobe(src_path)
            src_duration_s = src_duration_cache[src_key]

            vmaf_mean = (
                extract_vmaf_mean(vmaf_json_path) if vmaf_json_path.exists() else None
            )

            real_s, user_s, sys_s = parse_time_real_user_sys(ffmpeg_log)
            cpu_total_s = (
                (user_s + sys_s) if (user_s is not None and sys_s is not None) else None
            )

            # speed: duration / wall time  (x realtime)
            speed = (
                (src_duration_s / real_s)
                if (src_duration_s and real_s and real_s > 0)
                else None
            )

            rows.append(
                {
                    "codec": codec,
                    "input_type": input_type,
                    "preset": preset,
                    "src_path": str(src_path),
                    "out_path": str(mp4_path),
                    "src_size_bytes": src_size,
                    "out_size_bytes": out_size,
                    "size_ratio_out_over_src": size_ratio,
                    "src_duration_s": src_duration_s,
                    "vmaf_mean": vmaf_mean,
                    "real_time_s": real_s,
                    "speed": speed,
                    "cpu_user_s": user_s,
                    "cpu_sys_s": sys_s,
                    "cpu_total_s": cpu_total_s,
                    "ffmpeg_log_path": str(ffmpeg_log),
                    "vmaf_json_path": (
                        str(vmaf_json_path) if vmaf_json_path.exists() else None
                    ),
                }
            )

        df = pd.DataFrame(rows)

        if not df.empty:
            df["codec_family"] = df["codec"].map(
                lambda x: (
                    "h264"
                    if x == "libx264"
                    else (
                        "h265"
                        if x == "libx265"
                        else ("av1" if x in {"libsvtav1", "libaom-av1"} else "other")
                    )
                )
            )
            df = df.sort_values(
                ["codec_family", "codec", "input_type", "preset"]
            ).reset_index(drop=True)

        return df

    # ========== 実行 ==========
    df = collect_codec_benchmark_rows()
    df.to_csv(OUT_DIR / "codec_benchmark_summary.tsv", sep="\t", index=False)
    df
    return df, pd


@app.cell
def _(df, pd):
    import matplotlib.pyplot as plt
    import numpy as np

    # =========================================================
    # Plot from pandas DataFrame `df`
    # - Compare by encoder implementation (libx264 / libx265 / libsvtav1)
    # - Normalized preset axis: slow+K .. default .. fast+K
    # - Annotate each point with original preset value
    # - ASCII only (no Japanese)
    # =========================================================
    # ---- user knobs ----
    INPUT_TYPE = "compat"  # set None to include all input_type
    SAVE = True
    OUTDIR = "out"

    # AV1 implementations (larger preset => faster)
    AV1_IMPLS = {
        "libsvtav1",
        "libaom-av1",
        "librav1e",
    }

    TARGET_CODECS = ["libx264", "libx265", "libsvtav1"]

    # ---- helpers ----
    def _ensure_numeric(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        d = d.copy()
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        return d

    def _normalize_speed_rank(codec: str, presets_sorted: list[int]) -> dict[int, int]:
        """
        Return mapping preset -> speed_rank (higher = faster)
        """
        if codec in AV1_IMPLS:
            # larger preset is faster
            order = sorted(presets_sorted)  # slow .. fast
            return {p: i + 1 for i, p in enumerate(order)}
        else:
            # smaller preset is faster
            order = sorted(presets_sorted, reverse=True)  # slow .. fast
            return {p: i + 1 for i, p in enumerate(order)}

    def _build_speed_labels_for_codec(
        codec: str, codec_df: pd.DataFrame
    ) -> pd.DataFrame:
        d = codec_df.copy()
        presets = sorted(pd.unique(d["preset"].dropna()).tolist())
        if not presets:
            return d

        preset_to_rank = _normalize_speed_rank(codec, presets)
        d["speed_rank"] = d["preset"].map(preset_to_rank).astype("Int64")

        ranks = sorted(preset_to_rank.values())
        center_rank = ranks[len(ranks) // 2]

        def rank_to_label(r: int) -> str:
            if r == center_rank:
                return "default"
            if r < center_rank:
                return f"slow+{center_rank - r}"
            return f"fast+{r - center_rank}"

        d["speed_label"] = d["speed_rank"].apply(
            lambda x: rank_to_label(int(x)) if pd.notna(x) else None
        )

        max_offset = max(abs(int(r) - center_rank) for r in ranks)
        ordered_labels = (
            [f"slow+{k}" for k in range(max_offset, 0, -1)]
            + ["default"]
            + [f"fast+{k}" for k in range(1, max_offset + 1)]
        )

        d["speed_label_cat"] = pd.Categorical(
            d["speed_label"], categories=ordered_labels, ordered=True
        )
        d["speed_pos"] = d["speed_label_cat"].cat.codes.astype(float)
        return d

    def _prep_plot_df(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        if INPUT_TYPE is not None:
            d = d[d["input_type"] == INPUT_TYPE]

        d = d[d["codec"].isin(TARGET_CODECS)]

        d = _ensure_numeric(
            d,
            [
                "preset",
                "size_ratio_out_over_src",
                "vmaf_mean",
                "real_time_s",
                "cpu_total_s",
            ],
        )

        d = d.dropna(subset=["codec", "preset"])

        parts = []
        for codec, g in d.groupby("codec", sort=False):
            parts.append(_build_speed_labels_for_codec(codec, g))

        d2 = pd.concat(parts, ignore_index=True)
        d2 = d2.dropna(subset=["speed_label", "speed_pos"])
        return d2

    def _plot_dual_axis_by_impl(
        data: pd.DataFrame,
        y_left_col: str,
        y_right_col: str,
        title: str,
        left_ylabel: str,
        right_ylabel: str,
        out_png: str | None = None,
    ):
        fig, ax1 = plt.subplots(figsize=(11, 5.8))
        ax2 = ax1.twinx()

        # x-axis categories
        cats = data["speed_label_cat"].dtype.categories.tolist()
        x_ticks = np.arange(len(cats))
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(cats)

        impls = [c for c in TARGET_CODECS if c in set(data["codec"])]

        left_handles = []
        right_handles = []

        for impl in impls:
            d = data[data["codec"] == impl].copy()
            d = d.sort_values("speed_label_cat")

            x = d["speed_label_cat"].cat.codes.to_numpy()
            y1 = d[y_left_col].to_numpy()
            y2 = d[y_right_col].to_numpy()
            presets = d["preset"].astype(int).to_numpy()

            # Left axis (solid)
            (h1,) = ax1.plot(x, y1, marker="o", linestyle="-", label=f"{impl} left")
            left_handles.append(h1)

            # Annotate preset on left-axis points
            for xi, yi, p in zip(x, y1, presets):
                ax1.annotate(
                    f"p{p}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                )

            # Right axis (dashed, same color)
            (h2,) = ax2.plot(
                x,
                y2,
                marker="s",
                linestyle="--",
                color=h1.get_color(),
                label=f"{impl} right",
            )
            right_handles.append(h2)

            # Annotate preset on right-axis points (slightly lower)
            for xi, yi, p in zip(x, y2, presets):
                ax2.annotate(
                    f"p{p}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, -10),
                    ha="center",
                    fontsize=8,
                )

        ax1.set_title(title)
        ax1.set_xlabel("speed_label (normalized around default)")
        ax1.set_ylabel(left_ylabel)
        ax2.set_ylabel(right_ylabel)
        ax1.grid(True, which="both", axis="both")

        handles = left_handles + right_handles
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="best", frameon=True)

        fig.tight_layout()
        if out_png:
            fig.savefig(out_png, dpi=150)
        return fig

    # ---- main ----
    plot_df = _prep_plot_df(df)

    # 1) Compression efficiency vs VMAF
    fig1 = _plot_dual_axis_by_impl(
        data=plot_df.dropna(subset=["size_ratio_out_over_src", "vmaf_mean"]),
        y_left_col="size_ratio_out_over_src",
        y_right_col="vmaf_mean",
        title=f"Efficiency vs VMAF (by encoder impl) input_type={INPUT_TYPE}",
        left_ylabel="size_ratio (out/src)",
        right_ylabel="VMAF mean",
        out_png=f"{OUTDIR}/plot_efficiency_vmaf_by_impl.png" if SAVE else None,
    )

    # 2) Real time vs CPU total
    fig2 = _plot_dual_axis_by_impl(
        data=plot_df.dropna(subset=["speed", "cpu_total_s"]),
        y_left_col="speed",
        y_right_col="cpu_total_s",
        title=f"Speed vs CPU time (by encoder impl) input_type={INPUT_TYPE}",
        left_ylabel="speed (x realtime) = src_duration_s / real_time_s",
        right_ylabel="cpu_total_s (user+sys)",
        out_png=f"{OUTDIR}/plot_time_cpu_by_impl.png" if SAVE else None,
    )

    plt.show()

    if SAVE:
        print("saved:")
        print(f"- {OUTDIR}/plot_efficiency_vmaf_by_impl.png")
        print(f"- {OUTDIR}/plot_time_cpu_by_impl.png")
    return


if __name__ == "__main__":
    app.run()
