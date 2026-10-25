#!/usr/bin/env python3
"""Keep executable LLVM symbols private and adapt rustc's LTO options to ld64.lld."""

import os
import re
import sys


def linker_args(arguments):
    result = []
    for argument in arguments:
        if not argument.startswith("-Wl,") or "-plugin-opt=" not in argument:
            result.append(argument)
            continue
        options = []
        for option in argument[4:].split(","):
            if match := re.fullmatch(r"-plugin-opt=O([0-3])", option):
                level = match[1]
                options.extend([f"--lto-O{level}", f"--lto-CGO{level}"])
            elif option.startswith("-plugin-opt=mcpu="):
                options.extend(["-mcpu", option.removeprefix("-plugin-opt=mcpu=")])
            else:
                options.append(option)
        result.append("-Wl," + ",".join(options))
    # Build scripts are executables too. Preserve exports from proc-macro and
    # shared libraries, whose Rust entry points must remain visible to rustc.
    if not any(arg in ("-dynamiclib", "-shared", "-bundle") for arg in arguments):
        result.append("-Wl,-no_exported_symbols")
    return result


if __name__ == "__main__":
    arguments = sys.argv[1:]
    compiler = "clang"
    if arguments[:1] == ["--clang"]:
        compiler, arguments = arguments[1], arguments[2:]
    os.execvp(compiler, [compiler, *linker_args(arguments)])
