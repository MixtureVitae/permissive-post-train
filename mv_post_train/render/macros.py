"""The text-number macro files the paper loads, shared by every renderer.

Each renderer writes one file of LaTeX macros for the numbers its tables hold,
for example `\\ptwo{mv.sft-ot3.avg}` from `phase2_numbers.tex`. Every file has
the same form: a `\\providecommand` that prints a key's value, or the key in
bold between question marks if it does not exist, and one definition per key.
`macro_lines` writes that part, so every file can be read back the same way
(`parse_macros`), which `mvpt check-paper` does to compare the paper's copy
with the generated one key by key.
"""
from __future__ import annotations

import re
from pathlib import Path

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_word(n: int) -> str:
    """`n` in words for 0 to 99 ("twenty-four"), otherwise its digits."""
    if 0 <= n < 20:
        return _ONES[n]
    if 20 <= n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    return str(n)


def macro_lines(cmd: str, values: dict[str, str], *, sort: bool = False) -> list[str]:
    """The `\\providecommand` for `\\<cmd>` and one definition per key."""
    lines = [rf"\providecommand{{\{cmd}}}[1]{{\ifcsname {cmd}@#1\endcsname\csname {cmd}@#1"
             rf"\endcsname\else\textbf{{??#1??}}\fi}}"]
    keys = sorted(values) if sort else list(values)
    return lines + [f"\\expandafter\\def\\csname {cmd}@{k}\\endcsname{{{values[k]}}}" for k in keys]


_PROVIDE = re.compile(r"^\\providecommand\{\\(\w+)\}\[1\]")
_DEF = re.compile(r"^\\expandafter\\def\\csname (\w+)@(.+?)\\endcsname\{(.*)\}$")


def parse_macros(path: Path) -> tuple[str, dict[str, str]]:
    """(command, {key: value}) of a macro file written with `macro_lines`."""
    cmd, values = None, {}
    for line in Path(path).read_text().splitlines():
        if m := _PROVIDE.match(line):
            cmd = m.group(1)
        elif m := _DEF.match(line):
            values[m.group(2)] = m.group(3)
    if cmd is None:
        raise ValueError(f"{path} is not a macro file: no \\providecommand line.")
    return cmd, values


def rotated(label: str, longest: int = 14) -> str:
    """A section label turned on its side for a block of six table rows. A label
    longer than `longest` characters is stacked on two lines at its first space,
    so that it is no taller than the block. The parser of `paper_check` reads
    the stacked label as the one-line label."""
    if len(label) > longest and " " in label:
        first, rest = label.split(" ", 1)
        label = f"\\shortstack{{{first}\\\\ {rest}}}"
    return f"\\rotatebox{{90}}{{{label}}}"
