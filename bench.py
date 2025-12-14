import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from subprocess import run
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
        ("libx264", "compat", "1"),
        ("libx264", "compat", "3"),
        ("libx264", "compat", "5"), # default
        ("libx264", "compat", "7"),
        ("libx264", "compat", "9"),

        # ref: https://x265.readthedocs.io/en/stable/presets.html
        ("libx265", "compat", "1"),
        ("libx265", "compat", "3"),
        ("libx265", "compat", "5"), # default
        ("libx265", "compat", "7"),
        ("libx265", "compat", "9"),

        # ref: https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Docs/CommonQuestions.md#what-presets-do
        # Generally speaking, presets 1-3 represent extremely high efficiency, for use when encode time is not important and quality/size of the resulting video file is critical.
        # Presets 4-6 are commonly used by home enthusiasts as they represent a balance of efficiency and reasonable compute time.
        # Presets between 7 and 13 are used for fast and real-time encoding. One should use the lowest preset that is tolerable.
        ("libsvtav1", "compat", "4"),
        ("libsvtav1", "compat", "6"),
        ("libsvtav1", "compat", "8"), # default
        ("libsvtav1", "compat", "10"),
        ("libsvtav1", "compat", "12"),
    ]
    for codec, input_type, preset in cases:
        print(f"convert: {codec=} {input_type=} {preset=}")
        out_prefix = f"out/{codec}_{input_type}_preset{preset}"

        run(f"""
        {time_command} {ffmpeg_command} -y -i movies/{input_type}.mov \
            -vf "{video_filter}" \
            -c:v {codec} -preset {preset} \
            -an \
            {out_prefix}.mp4 > {out_prefix}_ffmpeg.log 2>&1""", shell=True, check=True)

        run(f"""
        {ffprobe_command} {out_prefix}.mp4 > {out_prefix}_ffprobe.txt 2>&1
        """, shell=True, check=True)

        run(f"""
        {ffmpeg_command} -i {out_prefix}.mp4 -i movies/{input_type}.mov \
            -lavfi "[0:v]{video_filter}[dist];[1:v]{video_filter}[ref];[dist][ref]libvmaf=log_fmt=json:log_path={out_prefix}_vmaf.json" \
            -f null - > {out_prefix}_vmaf.txt 2>&1
        """, shell=True, check=True)
    return


if __name__ == "__main__":
    app.run()
