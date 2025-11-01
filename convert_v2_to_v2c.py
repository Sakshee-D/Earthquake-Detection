import sys
import re
from pathlib import Path
from datetime import datetime

HEADER_DATA_MARKER_RE = re.compile(r"\b(\d+)\s+points\s+of\s+accel\s+data\b", re.IGNORECASE)
UNITS_LINE_RE = re.compile(r"in\s+cm/sec2", re.IGNORECASE)

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?")


def extract_floats_from_v2(text: str):
    """Extract floats from the COSMOS V2 data section.
    We look for the 'points of accel data ...' marker, then parse floats from subsequent lines.
    Returns (values:list[float], n_from_header:int|None)
    """
    lines = text.splitlines()
    data_started = False
    values = []
    n_from_header = None

    for i, line in enumerate(lines):
        if not data_started:
            m = HEADER_DATA_MARKER_RE.search(line)
            if m is not None:
                try:
                    n_from_header = int(m.group(1))
                except Exception:
                    n_from_header = None
                # Data usually begins on this line or next lines; continue to parse floats starting here
                data_started = True
                # Do not skip this line; parse it as well to catch trailing floats
        # Once data_started, parse every following line for floats
        if data_started:
            for tok in FLOAT_RE.findall(line):
                try:
                    # Skip obvious metadata like fixed-width format descriptor e.g. '(8f10.5)'
                    if tok.endswith('f10.5'):
                        continue
                except Exception:
                    pass
                try:
                    v = float(tok)
                    values.append(v)
                except ValueError:
                    continue
    return values, n_from_header


def write_v2c_like(out_path: Path, values: list[float], fs: float | None = None, units: str = "cm/s^2", per_line: int = 10):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("COSMOS Strong-Motion Data (Converted)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if fs is not None:
            f.write(f"Sampling rate (Hz): {fs}\n")
        f.write(f"Units: {units}\n")
        f.write("Note: Converted from COSMOS V2 to V2c-like plain text.\n\n")
        f.write(f"{len(values)} acceleration pts\n\n")
        # Write values in blocks
        for i in range(0, len(values), per_line):
            block = values[i:i+per_line]
            f.write(" ".join(f"{v:.6f}" for v in block) + "\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_v2_to_v2c.py <input.v2> <output.V2c> [fs] [units]")
        sys.exit(1)
    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    fs = float(sys.argv[3]) if len(sys.argv) >= 4 else None
    units = sys.argv[4] if len(sys.argv) >= 5 else "cm/s^2"

    text = inp.read_text(encoding="utf-8", errors="ignore")
    values, n_from_header = extract_floats_from_v2(text)

    # If header declared N points, try to truncate/exact match
    if n_from_header is not None and len(values) >= n_from_header:
        values = values[-n_from_header:]  # some files include parameter floats before data floats

    write_v2c_like(outp, values, fs=fs, units=units)
    print(f"Wrote {len(values)} samples to {outp}")


if __name__ == "__main__":
    main()
