#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

scmap.py — Read, scale, and write compressor maps (.fae-style) in Python.

 

Replicates the legacy Fortran behaviour:

  - Reads FREE or FIXED format (.fae-like) files

  - Applies global or per-speed-line scale factors

  - Resets stored scale factors to 1.0 in the written file (data are scaled)

  - Writes FIXED format by default (matches WTMAP)

 

FREE vs FIXED (as implemented from the original Fortran):

  Title line is always 10*A4 (40 chars). FREE mode then reads numbers list-directed.

  FIXED mode uses exact widths:

    Title: 10*A4 (40 chars)

    NS,NR: 2I2  (4 chars total)

    Scales: 8F8.4 (we only emit 4)

    SP: F8.0 per line

    PR/MF/EF lines: 10F8.3 with wrap

 

Author: Converted for Jamie Archer

"""

 

from dataclasses import dataclass

from typing import List, Sequence, Tuple, Optional, Union

import argparse

import sys

import io

import math

import os

# -----------------------------

# Data model

# -----------------------------

 

@dataclass

class MapData:

    title: List[str]          # 10 × 4-char chunks

    ns: int

    nr: int

    scalsp: float

    scalpr: float

    scalmf: float

    scalef: float

    sp: List[float]           # length ns

    pr: List[List[float]]     # ns × nr

    mf: List[List[float]]     # ns × nr

    ef: List[List[float]]     # ns × nr

# -----------------------------

# Utilities

# -----------------------------

 

def _pad_or_trim_title_chunks(chunks: Sequence[str]) -> List[str]:

    """Ensure exactly 10 chunks of width 4 each."""

    chunks = list(chunks)

    if len(chunks) < 10:

        chunks += [""] * (10 - len(chunks))

    chunks = chunks[:10]

    return [c[:4].ljust(4) for c in chunks]

 

def _readline_or_raise(f: io.TextIOBase) -> str:

    line = f.readline()

    if line == "":

        raise EOFError("Unexpected end-of-file while reading map.")

    return line

 

def _slice_fixed_fields(line: str, width: int) -> List[str]:

    return [line[i:i+width] for i in range(0, len(line), width)]

 

def _read_fixed_floats(f: io.TextIOBase, count: int, width: int = 8) -> List[float]:

    """Read exactly `count` fixed-width floats across as many lines as needed."""

    vals: List[float] = []

    while len(vals) < count:

        line = _readline_or_raise(f).rstrip("\n")

        if not line:

            continue

        fields = _slice_fixed_fields(line, width)

        for tok in fields:

            tok_str = tok.strip()

            if tok_str == "":

                continue

            vals.append(float(tok_str))

            if len(vals) == count:

                break

    return vals

 

def _read_free_floats(f: io.TextIOBase, count: int) -> List[float]:

    """List-directed reading style: read whitespace-separated floats until `count` collected."""

    vals: List[float] = []

    while len(vals) < count:

        line = _readline_or_raise(f)

        tokens = line.strip().split()

        for t in tokens:

            vals.append(float(t))

            if len(vals) == count:

                break

    return vals

 

def _parse_scaler_arg(arg: str, ns: int) -> List[float]:

    """

    Convert a scaler argument into a per-line list of length ns.

    Accepts either a single float or CSV of ns floats.

    """

    if arg is None:

        return [1.0] * ns

    s = arg.strip()

    if "," in s:

        parts = [p.strip() for p in s.split(",") if p.strip() != ""]

        if len(parts) != ns:

            raise ValueError(f"Expected {ns} comma-separated values, got {len(parts)}")

        return [float(p) for p in parts]

    # scalar

    return [float(s)] * ns

 

# -----------------------------

# Reading

# -----------------------------

 

def read_title_fixed(f: io.TextIOBase) -> List[str]:

    line = _readline_or_raise(f).rstrip("\n")

    if len(line) < 40:

        line = line.ljust(40)

    chunks = _slice_fixed_fields(line[:40], 4)

    return _pad_or_trim_title_chunks(chunks)

 

def read_map_fixed(path: str) -> MapData:

    with open(path, "r", encoding="utf-8", errors="replace") as f:

        title = read_title_fixed(f)

 

        # NS, NR from 2I2 (4 chars total)

        line = _readline_or_raise(f).rstrip("\n")

        line = line.ljust(4)

        ns = int(line[0:2])

        nr = int(line[2:4])

 

        # Scale factors: read the next line (we expect at least 32 chars; 4 floats of width 8)

        line = _readline_or_raise(f).rstrip("\n")

        line = line.ljust(32)

        scalsp = float(line[0:8])

        scalpr = float(line[8:16])

        scalmf = float(line[16:24])

        scalef = float(line[24:32])

 

        sp: List[float] = []

        pr: List[List[float]] = []

        mf: List[List[float]] = []

        ef: List[List[float]] = []

 

        for _ in range(ns):

            # speed line F8.0

            sl = _readline_or_raise(f).rstrip("\n")

            sl = sl.ljust(8)

            sp.append(float(sl[:8]))

 

            # PR / MF / EF rows: each NR floats, width 8, wrapped 10 per line

            pr.append(_read_fixed_floats(f, nr, width=8))

            mf.append(_read_fixed_floats(f, nr, width=8))

            ef.append(_read_fixed_floats(f, nr, width=8))

 

        return MapData(title, ns, nr, scalsp, scalpr, scalmf, scalef, sp, pr, mf, ef)

 

def read_map_free(path: str) -> MapData:

    with open(path, "r", encoding="utf-8", errors="replace") as f:

        # Title still 10*A4

        title = read_title_fixed(f)

 

        # NS NR list-directed

        line = _readline_or_raise(f)

        parts = line.strip().split()

        if len(parts) < 2:

            # try reading an additional line if split across

            line2 = _readline_or_raise(f)

            parts = (line.strip() + " " + line2.strip()).split()

        ns, nr = int(parts[0]), int(parts[1])

 

        # four scale factors (list-directed)

        line = _readline_or_raise(f)

        scales = [float(x) for x in line.strip().split()]

        # if someone split across lines, keep reading

        while len(scales) < 4:

            scales.extend([float(x) for x in _readline_or_raise(f).strip().split()])

        scalsp, scalpr, scalmf, scalef = scales[:4]

 

        sp: List[float] = []

        pr: List[List[float]] = []

        mf: List[List[float]] = []

        ef: List[List[float]] = []

 

        for _ in range(ns):

            # Speed (list-directed)

            floats = []

            while not floats:

                floats = [float(x) for x in _readline_or_raise(f).strip().split()]

            sp.append(floats[0])

 

            pr.append(_read_free_floats(f, nr))

            mf.append(_read_free_floats(f, nr))

            ef.append(_read_free_floats(f, nr))

 

        return MapData(title, ns, nr, scalsp, scalpr, scalmf, scalef, sp, pr, mf, ef)

 

def read_map(path: str, read_format: str = "auto") -> MapData:

    """

    read_format: 'auto' | 'fixed' | 'free'

    Auto: try fixed, then free.

    """

    if read_format == "fixed":

        return read_map_fixed(path)

    if read_format == "free":

        return read_map_free(path)

    # auto

    try:

        return read_map_fixed(path)

    except Exception:

        return read_map_free(path)

 

# -----------------------------

# Writing

# -----------------------------

 

def write_title_fixed(f: io.TextIOBase, title_chunks: Sequence[str]) -> None:

    chunks = _pad_or_trim_title_chunks(title_chunks)

    f.write("".join(c[:4].ljust(4) for c in chunks) + "\n")

 

def write_map_fixed(path: str, m: MapData) -> None:

    with open(path, "w", encoding="utf-8") as f:

        # Title 10A4

        write_title_fixed(f, m.title)

        # 2I2 for NS, NR (no separator)

        f.write(f"{m.ns:2d}{m.nr:2d}\n")

        # four scales as F8.4 (emit 4)

        f.write(f"{m.scalsp:8.4f}{m.scalpr:8.4f}{m.scalmf:8.4f}{m.scalef:8.4f}\n")

 

        for lo in range(m.ns):

            # speed as F8.0

            f.write(f"{m.sp[lo]:8.0f}\n")

 

            def out_row(row: List[float]) -> None:

                for i in range(0, len(row), 10):

                    chunk = row[i:i+10]

                    f.write("".join(f"{x:8.3f}" for x in chunk) + "\n")

 

            out_row(m.pr[lo])

            out_row(m.mf[lo])

            out_row(m.ef[lo])

 

def write_map_free(path: str, m: MapData) -> None:

    """

    'Free' writer: still writes title as 10*A4 (to remain compatible with legacy reads),

    then whitespace-separated numerics with conventional precision and wrap at 10 per line.

    """

    with open(path, "w", encoding="utf-8") as f:

        write_title_fixed(f, m.title)

        f.write(f"{m.ns} {m.nr}\n")

        f.write(f"{m.scalsp:.4f} {m.scalpr:.4f} {m.scalmf:.4f} {m.scalef:.4f}\n")

        for lo in range(m.ns):

            f.write(f"{m.sp[lo]:.0f}\n")

            def out_row(row: List[float]) -> None:

                for i in range(0, len(row), 10):

                    f.write(" ".join(f"{x:.3f}" for x in row[i:i+10]) + "\n")

            out_row(m.pr[lo])

            out_row(m.mf[lo])

            out_row(m.ef[lo])

 

def write_map(path: str, m: MapData, write_format: str = "fixed") -> None:

    if write_format == "fixed":

        write_map_fixed(path, m)

    elif write_format == "free":

        write_map_free(path, m)

    else:

        raise ValueError("write_format must be 'fixed' or 'free'")

 

# -----------------------------

# Scaling

# -----------------------------

 

def scale_map(

    m: MapData,

    scalsp_per_line: Sequence[float],

    scalpr_per_line: Sequence[float],

    scalmf_per_line: Sequence[float],

    scalef_per_line: Sequence[float],

) -> MapData:

    """

    Apply scale factors in-place to SP, PR, MF, EF.

    Each scale sequence must have length ns.

    """

    if not (len(scalsp_per_line) == len(scalpr_per_line) == len(scalmf_per_line) == len(scalef_per_line) == m.ns):

        raise ValueError("All per-line scale sequences must have length ns")

 

    for lo in range(m.ns):

        spf = scalsp_per_line[lo]

        prf = scalpr_per_line[lo]

        mff = scalmf_per_line[lo]

        eff = scalef_per_line[lo]

 

        m.sp[lo] = m.sp[lo] * spf

        m.pr[lo] = [x * prf for x in m.pr[lo]]

        m.mf[lo] = [x * mff for x in m.mf[lo]]

        m.ef[lo] = [x * eff for x in m.ef[lo]]

 

    # Match Fortran behaviour: after scaling the data, reset stored scale factors to 1.0

    m.scalsp = 1.0

    m.scalpr = 1.0

    m.scalmf = 1.0

    m.scalef = 1.0

 

    return m

 

# -----------------------------

# CLI

# -----------------------------

 

def main(argv: Optional[List[str]] = None) -> int:

    p = argparse.ArgumentParser(

        description="Scale compressor map files (.fae-style) in Python (FREE/FIXED)."

    )

    p.add_argument("input", help="Input .fae (text) file")

    p.add_argument("output", help="Output .fae (text) file")

    p.add_argument("--read-format", choices=["auto", "fixed", "free"], default="auto",

                   help="How to read the input file (default: auto-detect)")

    p.add_argument("--write-format", choices=["fixed", "free"], default="fixed",

                   help="How to write the output file (default: fixed, matches legacy WTMAP)")

 

    # Scaling: either scalar or CSV with ns entries

    p.add_argument("--scalsp", default="1.0",

                   help="Scale factor(s) for SPEED. Scalar (e.g., 1.0) or CSV per speed line.")

    p.add_argument("--scalpr", default="1.0",

                   help="Scale factor(s) for PRESSURE RATIO. Scalar or CSV per speed line.")

    p.add_argument("--scalmf", default="1.0",

                   help="Scale factor(s) for MASS FLOW. Scalar or CSV per speed line.")

    p.add_argument("--scalef", default="1.0",

                   help="Scale factor(s) for EFFICIENCY. Scalar or CSV per speed line.")

 

    args = p.parse_args(argv)

 

    m = read_map(args.input, read_format=args.read_format)

 

    # Interpret scale arguments (may depend on ns)

    scalsp_per_line = _parse_scaler_arg(args.scalsp, m.ns)

    scalpr_per_line = _parse_scaler_arg(args.scalpr, m.ns)

    scalmf_per_line = _parse_scaler_arg(args.scalmf, m.ns)

    scalef_per_line = _parse_scaler_arg(args.scalef, m.ns)

 

    m = scale_map(m, scalsp_per_line, scalpr_per_line, scalmf_per_line, scalef_per_line)

    write_map(args.output, m, write_format=args.write_format)

 

    print(f"Scaled map written to: {args.output}")

    return 0

 

if __name__ == "__main__":

    sys.exit(main())

 
