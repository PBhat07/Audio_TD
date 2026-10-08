import os
import re

import matplotlib.pyplot as plt


GPU_PATTERN = re.compile(
    r"GPU MEMORY \| "
    r"stage=(?P<stage>\S+) \| "
    r"allocated=(?P<allocated>[\d.]+) MiB \| "
    r"reserved=(?P<reserved>[\d.]+) MiB \| "
    r"peak_allocated=(?P<peak_allocated>[\d.]+) MiB \| "
    r"peak_reserved=(?P<peak_reserved>[\d.]+) MiB \| "
    r"free=(?P<free>[\d.]+) MiB \| "
    r"total=(?P<total>[\d.]+) MiB"
)


def parse_gpu_memory_log(log_path: str) -> list[dict]:
    """Parse GPU memory measurements from the pipeline log."""

    records = []

    if not os.path.exists(log_path):
        return records

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = GPU_PATTERN.search(line)

            if not match:
                continue

            data = match.groupdict()

            records.append(
                {
                    "stage": data["stage"],
                    "allocated": float(data["allocated"]),
                    "reserved": float(data["reserved"]),
                    "peak_allocated": float(data["peak_allocated"]),
                    "peak_reserved": float(data["peak_reserved"]),
                    "free": float(data["free"]),
                    "total": float(data["total"]),
                }
            )

    return records
def generate_gpu_memory_report(
    log_path: str = "logs/gpu_memory.log",
    output_path: str = "logs/gpu_memory_report.png",
) -> None:

    records = parse_gpu_memory_log(log_path)

    if not records:
        return

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    headers = [
        "Stage",
        "Allocated",
        "Reserved",
        "Peak Allocated",
        "Peak Reserved",
        "Free",
    ]

    rows = []

    for record in records:
        rows.append(
            [
                record["stage"],
                f'{record["allocated"]:.0f} MB',
                f'{record["reserved"]:.0f} MB',
                f'{record["peak_allocated"]:.0f} MB',
                f'{record["peak_reserved"]:.0f} MB',
                f'{record["free"]:.0f} MB',
            ]
        )

    fig, ax = plt.subplots(
        figsize=(14, 1.2 + len(rows) * 0.45)
    )

    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    ax.set_title(
        "GPU Memory Report — Most Recent Pipeline Execution",
        fontsize=14,
        pad=20,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)