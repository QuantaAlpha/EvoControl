import ctypes
import json
import logging
import re
import resource
import threading
import time
from functools import cache
from pathlib import Path
from typing import Any

from tqdm import tqdm

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------

# Define ANSI color codes
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[31;1m",  # Bright Red
    "DEBUG": "\033[36m",  # Cyan
    "TIME": "\033[37;2m",  # Dim White
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname

        # Add colors based on level
        level_color = COLORS.get(levelname, COLORS["RESET"])

        # Format the message with colors
        record.levelname_colored = f"{level_color}{levelname:<8}{COLORS['RESET']}"
        record.time_colored = f"{COLORS['TIME']}{self.formatTime(record)}{COLORS['RESET']}"

        return super().format(record)


def setup_logger():
    """Configure logging with color formatter."""
    handler = logging.StreamHandler()
    formatter = ColorFormatter(fmt="%(time_colored)s %(levelname_colored)s %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    # Remove existing handlers and add our custom one
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


# Registry of all programming languages for reference
LANGUAGE_REGISTRY = {
    "cpp": {"id": 0, "verbose_name": "C++", "md_langs": ["cpp", "c++"]},
    "java": {"id": 1, "verbose_name": "Java", "md_langs": ["java"]},
    "python": {"id": 2, "verbose_name": "Python", "md_langs": ["python", "py"]},
    "mysql": {"id": 3, "verbose_name": "MySQL", "md_langs": ["mysql", "sql"]},
    "c": {"id": 4, "verbose_name": "C", "md_langs": ["c"]},
    "csharp": {"id": 5, "verbose_name": "C#", "md_langs": ["cs", "csharp", "c#"]},
    "javascript": {
        "id": 6,
        "verbose_name": "JavaScript",
        "md_langs": ["js", "javascript", "node"],
    },
    "ruby": {
        "id": 7,
        "verbose_name": "Ruby",
        "md_langs": ["rb", "ruby", "jruby", "macruby", "rake", "rbx"],
    },
    "bash": {
        "id": 8,
        "verbose_name": "Bash",
        "md_langs": ["sh", "bash", "shell", "shell-script", "zsh"],
    },
    "swift": {"id": 9, "verbose_name": "Swift", "md_langs": ["swift"]},
    "golang": {"id": 10, "verbose_name": "Go", "md_langs": ["golang", "go"]},
    "python3": {"id": 11, "verbose_name": "Python3", "md_langs": ["python", "py"]},
    "scala": {"id": 12, "verbose_name": "Scala", "md_langs": ["scala"]},
    "kotlin": {"id": 13, "verbose_name": "Kotlin", "md_langs": ["kotlin"]},
    "mssql": {"id": 14, "verbose_name": "MS SQL Server", "md_langs": ["tsql", "mssql"]},
    "oraclesql": {
        "id": 15,
        "verbose_name": "Oracle",
        "md_langs": ["plsql", "oraclesql"],
    },
    "rust": {"id": 18, "verbose_name": "Rust", "md_langs": ["rust", "rs"]},
    "php": {"id": 19, "verbose_name": "PHP", "md_langs": ["php"]},
    "typescript": {
        "id": 20,
        "verbose_name": "TypeScript",
        "md_langs": ["ts", "typescript"],
    },
    "racket": {"id": 21, "verbose_name": "Racket", "md_langs": ["racket"]},
    "erlang": {"id": 22, "verbose_name": "Erlang", "md_langs": ["erlang"]},
    "elixir": {"id": 23, "verbose_name": "Elixir", "md_langs": ["elixir"]},
    "dart": {"id": 24, "verbose_name": "Dart", "md_langs": ["dart"]},
    "pythondata": {
        "id": 25,
        "verbose_name": "Pandas",
        "md_langs": ["pandas", "pythondata"],
    },
    "react": {"id": 26, "verbose_name": "React", "md_langs": ["jsx", "react"]},
    "vanillajs": {
        "id": 27,
        "verbose_name": "Vanilla JS",
        "md_langs": ["js", "javascript", "vanillajs"],
    },
    "postgresql": {
        "id": 28,
        "verbose_name": "PostgreSQL",
        "md_langs": ["postgres", "postgresql", "pgsql"],
    },
    "cangjie": {"id": 29, "verbose_name": "Cangjie", "md_langs": ["cangjie"]},
}


def parse_range(range_str: str | None) -> tuple[int, int]:
    """Parse a range string in the format 'a:b' to get start and end values.

    If 'a' is omitted, start is 0.
    If 'b' is omitted, end is a very large number.

    Args:
        range_str: A string in format 'a:b' where a and b are integers

    Returns:
        A tuple (start, end) with the parsed range values

    Raises:
        ValueError: If the range format is invalid
    """
    if not range_str:
        return 0, int(1e9)

    parts = range_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Range format should be 'a:b', got '{range_str}'")

    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else int(1e9)

    return start, end


@cache
def get_md_lang(lang: str) -> str | None:
    """
    Returns the first Markdown code block identifier for the given language key from LANG_LOOKUP.
    Returns None if not found.
    """
    md_langs = LANGUAGE_REGISTRY.get(lang, {}).get("md_langs", [])
    return md_langs[0] if md_langs else None


@cache
def get_lang_by_md_lang(md_lang: str) -> str | None:
    """
    Returns the language key for the given Markdown code block identifier from LANG_LOOKUP.
    Returns None if not found.
    """
    if md_lang in LANGUAGE_REGISTRY["python3"]["md_langs"]:
        return "python3"
    return next(
        (key for key, value in LANGUAGE_REGISTRY.items() if md_lang in value["md_langs"]),
        None,
    )


@cache
def get_lang_by_verbose_name(verbose_name: str) -> str | None:
    """
    Returns the language key for the given verbose name from LANG_LOOKUP.
    Returns None if not found.
    """
    return next(
        (key for key, value in LANGUAGE_REGISTRY.items() if verbose_name.lower() == value["verbose_name"].lower()),
        None,
    )


# Standard imports for each supported language for competitive programming
# DO NOT expose this outside of this file. Use EFFIBENCH_REGISTRY instead.
_LANGUAGE_IMPORTS = {
    "python3": r"""
import re
from re import match, search, sub, split, findall, finditer
import sys
from sys import maxsize, stdin
import json
from json import loads
import math
from math import floor, ceil, factorial, sqrt, isqrt, inf, log2, log10, sin, cos, tan, pi, e, comb, perm, gcd, lcm
import copy
import pickle
import heapq
from heapq import heappush, heappop, heapify, heappushpop, nlargest, nsmallest
import bisect
from bisect import bisect_left, bisect_right
import string
from string import ascii_letters, ascii_lowercase, ascii_uppercase, digits, whitespace, punctuation, hexdigits
import random
import operator
import itertools
from itertools import combinations, permutations, product, groupby, chain, accumulate, zip_longest
import functools
from functools import lru_cache, cache, reduce
import collections
from collections import OrderedDict, defaultdict, Counter, deque
from typing import Set, Dict, List, Optional, Tuple

import sortedcontainers # pip install sortedcontainers
from sortedcontainers import SortedList, SortedDict, SortedSet
""",
    "java": r"""
import java.io.*;
import java.math.*;
import java.text.*;
import java.util.*;
import java.util.stream.*;
import java.util.function.*;
""",
    "javascript": r"""
const util = require('util');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert');
const os = require('os');
const http = require('http');
const https = require('https');
const url = require('url');
const querystring = require('querystring');
const zlib = require('zlib');
const stream = require('stream');
const buffer = require('buffer');
const events = require('events');
const child_process = require('child_process');
const readline = require('readline');
const process = require('process');
const string_decoder = require('string_decoder');
const timers = require('timers');
const perf_hooks = require('perf_hooks');
const dgram = require('dgram');  // UDP
const dns = require('dns');
const net = require('net');  // TCP
const tls = require('tls');
const vm = require('vm');

const _ = require('lodash');
const { PriorityQueue, Queue, Deque } = require('datastructures-js');
""",
    "cpp": r"""
// #include <bits/stdc++.h>
#include <iostream>
#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <deque>
#include <forward_list>
#include <functional>
#include <iomanip>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <list>
#include <bitset>
#include <fstream>
#include <sstream>
#include <iterator>
#include <random>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <utility>
#include <climits>
#include <tuple>
#include <cstdlib>
#include <cctype>

using namespace std;
""",
    "golang": r"""
package main

import (
	"io"
	"os"
	"fmt"
	"math"
	"sort"
	"time"
	"bufio"
	"regexp"
	"reflect"
	"strings"
	"strconv"
	"math/big"
	"math/bits"
	"math/rand"
	"container/heap"
	"container/list"
)
""",
    "ruby": r"""
require 'set'
require 'date'
require 'time'
require 'stringio'
require 'bigdecimal'
require 'securerandom'

require 'json'
require 'algorithms'
""",
    "rust": r"""
use std::mem;
use std::fmt::{self, Display};
use std::iter::FromIterator;
use std::io::{self, Read, Write};
use std::cmp::{min, max, Ordering};
use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque, BinaryHeap};
""",
}

# A subset of languages that are supported by EffiBench, with more information
EFFIBENCH_REGISTRY = {
    "cpp": {
        "id": 0,
        "verbose_name": "C++",
        "md_langs": ["cpp", "c++"],
        "llm_sandbox_lang": "cpp",
        "packages": [],
        "imports": _LANGUAGE_IMPORTS["cpp"],
        "flags": ["-O2", "-fsanitize=address"],
        "image": "gcc:14.2.0-bookworm",
        "executable": "g++",
    },
    "java": {
        "id": 1,
        "verbose_name": "Java",
        "md_langs": ["java"],
        "llm_sandbox_lang": "java",
        "packages": [],
        "imports": _LANGUAGE_IMPORTS["java"],
        "image": "openjdk:21-jdk-bookworm",
        "executable": "java",
    },
    "javascript": {
        "id": 6,
        "verbose_name": "JavaScript",
        "md_langs": ["js", "javascript", "node"],
        "llm_sandbox_lang": "javascript",
        "packages": ["lodash", "datastructures-js"],
        "imports": _LANGUAGE_IMPORTS["javascript"],
        "flags": ["--harmony"],
        "image": "node:22.14.0-bookworm",
        "executable": "node",
    },
    "ruby": {
        "id": 7,
        "verbose_name": "Ruby",
        "md_langs": ["rb", "ruby", "jruby", "macruby", "rake", "rbx"],
        "llm_sandbox_lang": "ruby",
        "packages": ["json", "algorithms"],
        "imports": _LANGUAGE_IMPORTS["ruby"],
        "flags": [],
        "image": "ruby:3.2.7-bookworm",
        "executable": "ruby",
    },
    "golang": {
        "id": 10,
        "verbose_name": "Go",
        "md_langs": ["golang", "go"],
        "llm_sandbox_lang": "go",
        "packages": [],
        "imports": _LANGUAGE_IMPORTS["golang"],
        "flags": [],
        "image": "golang:1.23.7-bookworm",
        "executable": "go",
    },
    "python3": {
        "id": 11,
        "verbose_name": "Python3",
        "md_langs": ["python", "py"],
        "llm_sandbox_lang": "python",
        "packages": ["sortedcontainers", "numpy"],
        "imports": _LANGUAGE_IMPORTS["python3"],
        "flags": [],
        "image": "python:3.11.11-bookworm",
        "executable": "python3",
    },
}

# Languages supported by EffiBench
EFFIBENCH_LANGS = list(EFFIBENCH_REGISTRY.keys())


@cache
def get_sandbox_lang(lang: str) -> str | None:
    """
    Returns the sandbox language for the given language key.
    Returns None if not found.
    """
    lang_info = EFFIBENCH_REGISTRY.get(lang)
    if lang_info:
        return lang_info.get("llm_sandbox_lang")
    return None


@cache
def get_all_sandbox_langs() -> list[str]:
    """
    Returns all sandbox languages.
    """
    return [lang_info["llm_sandbox_lang"] for lang_info in EFFIBENCH_REGISTRY.values()]


def prune_package_imports(code: str, lang: str) -> str:
    """
    Prunes all import statements from code and additionally 'package main' for Go.

    Args:
        code: Source code string to process
        lang: Programming language identifier (e.g., "python3", "golang")

    Returns:
        Code string with all imports removed

    Raises:
        ValueError: If language is not supported
    """
    if lang not in EFFIBENCH_REGISTRY:
        raise ValueError(f"Language '{lang}' is not supported. Supported languages: {EFFIBENCH_LANGS}")

    # Handle empty code
    if not code.strip():
        return ""

    # Define language-specific import patterns and special processing rules
    patterns = {
        "python3": [r"^\s*import\s+.*$", r"^\s*from\s+\w+.*\s+import.*$"],
        "javascript": [
            r"^\s*import\s+.*$",
            r"^\s*(const|let|var)\s+.*?=\s*require\(.*$",
            r"^\s*require\(.*$",
        ],
        "java": [r"^\s*import\s+.*$"],
        "cpp": [
            r"^\s*#include\s+.*$",
            r"^\s*using\s+namespace\s+.*$",
            r"^\s*using\s+\w+::.*$",
        ],
        "ruby": [r"^\s*(require|require_relative)\s+.*$"],
        "golang": [r"^\s*import\s+.*$", r"^\s*package\s+main\s*$"],
    }

    lines = code.splitlines()
    result_lines = []

    # Special processing for multi-line imports and blocks
    skip_until_pattern = None  # Use this to skip lines in multi-line blocks
    skip_current_line = False
    custom_go_package = None  # Store Go custom package if found

    # First pass: process all lines
    for i, line in enumerate(lines):
        skip_current_line = False

        # Handle multi-line skipping (for parenthesized imports)
        if skip_until_pattern:
            if re.search(skip_until_pattern, line):
                skip_until_pattern = None
            continue

        # Language-specific special processing
        if lang == "python3":
            # Handle Python's multi-line imports with parentheses
            if re.match(r"^\s*from\s+\w+.*\s+import\s+\(", line) and ")" not in line:
                skip_until_pattern = r"\)"
                skip_current_line = True

        elif lang == "golang":
            # Handle Go's custom package and import blocks
            if re.match(r"^\s*package\s+([a-zA-Z0-9_]+)\s*$", line):
                match = re.match(r"^\s*package\s+([a-zA-Z0-9_]+)\s*$", line)
                if match and match.group(1) != "main":
                    custom_go_package = line
                skip_current_line = True

            if re.match(r"^\s*import\s*\($", line) or line.strip() == "import (":
                skip_until_pattern = r"^\s*\)\s*$"
                skip_current_line = True

        # Check against language-specific patterns
        for pattern in patterns.get(lang, []):
            if re.match(pattern, line):
                skip_current_line = True
                break

        # Special case for JavaScript comments (preserve them)
        if lang == "javascript" and re.match(r"^\s*(//|/\*)", line):
            result_lines.append(line)
            continue

        # Add non-import lines to result
        if not skip_current_line:
            result_lines.append(line)

    # Add Go custom package if found
    if lang == "golang" and custom_go_package:
        result_lines.insert(0, custom_go_package)

    # Join the pruned lines
    pruned_code = "\n".join(result_lines)

    # Clean up blank lines
    pruned_code = re.sub(r"\n{3,}", "\n\n", pruned_code)  # No more than 2 consecutive newlines
    pruned_code = pruned_code.strip()

    return pruned_code


def postprocess_test_runner(test_runner: str, lang: str) -> str:
    # Prune import statements from the test runner
    test_runner = prune_package_imports(test_runner, lang)
    # Regularize the code submission placeholder
    lines = test_runner.split("\n")
    for i, line in enumerate(lines):
        if "==Code Submission==" in line:
            lines[i] = "==Code Submission=="
    test_runner = "\n".join(lines)
    return test_runner


def postprocess_solution(solution: str, lang: str) -> str:
    # Prune import statements from the solution
    solution = prune_package_imports(solution, lang)
    return solution


def find_first_public_class_name(code: str) -> str | None:
    """Find the name of the first public class in Java code.

    Args:
        code: Java source code

    Returns:
        The name of the first public class found, or None if no public class is found
    """
    public_class_matches = list(re.finditer(r"public\s+class\s+(\w+)", code))

    if not public_class_matches:
        return None  # No public classes found

    # Get the first public class
    first_match = public_class_matches[0]
    class_name = first_match.group(1)

    return class_name


def rename_java_class_to_main(code: str) -> str:
    """Rename the public class in Java code to Main.

    If there are multiple public classes, it renames the first one found.

    Args:
        code: Java source code

    Returns:
        Modified Java code with the public class renamed to Main
    """
    original_class_name = find_first_public_class_name(code)

    if not original_class_name:
        return code  # No public classes found

    if original_class_name == "Main":
        return code  # Already named Main

    # Replace the class name with Main
    code = re.sub(
        r"public\s+class\s+" + re.escape(original_class_name) + r"\b",
        "public class Main",
        code,
    )

    # Replace any constructor declarations
    code = re.sub(r"\b" + re.escape(original_class_name) + r"\s*\(", "Main(", code)

    # Replace any references to the class name as a type (variable declarations)
    code = re.sub(r"\b" + re.escape(original_class_name) + r"\s+([a-zA-Z0-9_]+)", r"Main \1", code)

    # Replace any "new ClassName" instantiations
    code = re.sub(r"new\s+" + re.escape(original_class_name) + r"\b", "new Main", code)

    # Replace any static references to the class name
    code = re.sub(r"\b" + re.escape(original_class_name) + r"\.", "Main.", code)

    return code


def get_full_code(lang: str, solution: str, test_runner: str) -> str:
    test_runner = postprocess_test_runner(test_runner, lang)
    solution = postprocess_solution(solution, lang)
    code = (
        EFFIBENCH_REGISTRY.get(lang, {}).get("imports", "")
        + "\n\n"
        + test_runner.replace("==Code Submission==", solution)
    )

    return code


def extract_code_blocks(text: str) -> list[dict[str, str]]:
    _CODE_BLOCK_PATTERN = re.compile(r"```([\w+-]*)(?:\n|\r\n)?(.*?)```", re.DOTALL)
    blocks: list[dict[str, str]] = []
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        lang = match.group(1).strip()
        code = match.group(2).strip()
        blocks.append({"lang": lang, "code": code})
    return blocks


def load_json_file(path: Path | str):
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {path}: {e}")
        raise
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except PermissionError:
        logging.error(f"Permission denied when reading {path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading {path}: {e}")
        raise


def _sanitize_for_json(obj: Any):
    import math

    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if np is not None and isinstance(obj, (np.floating, np.integer)):
        return _sanitize_for_json(obj.item())
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_json_file(path: Path | str, data, indent=4):
    path = Path(path)
    clean = _sanitize_for_json(data)
    path.write_text(json.dumps(clean, indent=indent, allow_nan=False))


def retry(func=None, max_retries=6, backoff_factor=2, error_types=(Exception,)):
    """
    Retry decorator with exponential backoff.
    Can be used as @retry or with parameters @retry(max_retries=3)
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return f(*args, **kwargs)
                except error_types as e:
                    if attempt == max_retries - 1:
                        raise e
                    retry_time = backoff_factor**attempt
                    print(
                        f"🟡 Error on attempt {attempt + 1}/{max_retries}: {str(e)}, retrying in {retry_time} seconds",
                        flush=True,
                    )
                    time.sleep(retry_time)

        return wrapper

    # Handle both @retry and @retry(...) cases
    if func is None:
        return decorator
    return decorator(func)


def execute_with_timeout(func, timeout, *args, **kwargs):
    """Execute a function with a timeout and terminate the thread if it exceeds the timeout.

    Args:
        func: The function to execute
        timeout: Maximum execution time in seconds
        *args, **kwargs: Arguments to pass to the function

    Returns:
        The result of the function

    Raises:
        TimeoutError: If the function execution exceeds the timeout
    """
    result = [None]
    exception = [None]

    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Terminate the thread using PyThreadState_SetAsyncExc
        tid = thread.ident
        # res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(KeyboardInterrupt))
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(SystemExit))
        if res != 1:
            # Handle error by clearing exception state
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        thread.join(0.1)  # Brief wait for termination
        raise TimeoutError(f"Timeout ({timeout}s) executing {func.__name__}")

    if exception[0]:
        raise exception[0]

    return result[0]


def create_logger(*args: Any, **kwargs: Any):
    """Creates a simple logger function using click.echo."""
    import click

    def log(msg: str, *extra_args: Any, **style_kwargs: Any) -> None:
        parts = []
        if args:
            parts.append(":".join(map(str, args)))
        if kwargs:
            parts.append(":".join(f"{k}:{v}" for k, v in kwargs.items()))
        if extra_args:
            parts.append(":".join(map(str, extra_args)))
        parts.append(str(msg))
        click.echo(click.style(":".join(parts), **style_kwargs), color=bool(style_kwargs))

    return log


def materialize_function_from_code(code: str, function_name: str, inject_imports: bool = True):
    if inject_imports:
        code = EFFIBENCH_REGISTRY["python3"]["imports"] + "\n\n" + code

    ns = {}
    try:
        exec(code, ns, ns)
    except Exception as e:
        raise ValueError(f"Error executing code: {e}")
    if function_name not in ns:
        raise ValueError(f"Function '{function_name}' not found in the code")
    return ns[function_name]


def try_int(s):
    try:
        return int(s)
    except ValueError:
        return s


def sort_problem_files(files: list[Path]) -> list[Path]:
    """Sort files by problem ID, handling both formats with and without source prefixes.

    For files named like "source_id_slug.ext" or "id_slug.ext", this ensures numeric IDs
    are sorted numerically rather than lexicographically.

    Args:
        files: List of Path objects to sort

    Returns:
        Sorted list of Path objects
    """
    return sorted(files, key=lambda x: tuple(try_int(part) for part in x.stem.split("_")))


def generate_problem_key(problem_id, title_slug, source=None) -> str:
    """Generate a standardized file key for a problem.

    Args:
        problem_id: The problem ID
        title_slug: The title slug (can be None)
        source: The source platform (e.g., "leetcode", "aizu") to prefix

    Returns:
        A standardized key in format "source_problem_id_title_slug" if source is provided,
        otherwise "problem_id_title_slug" or just "problem_id" if title_slug is None
    """
    if source:
        return f"{source}_{problem_id}_{title_slug}" if title_slug else f"{source}_{problem_id}"
    else:
        return f"{problem_id}_{title_slug}" if title_slug else str(problem_id)


def get_problem_key_from_file(file_path: Path) -> tuple[str, str | None]:
    """Extract problem_id and title_slug from a file path.

    Args:
        file_path: Path object representing the file

    Returns:
        A tuple of (problem_id, title_slug)
    """
    splits = file_path.stem.split("_", 2)
    # Handle source_id_slug format (e.g., "leetcode_123_two-sum")
    if len(splits) >= 3:
        return splits[1], splits[2]
    # Handle id_slug format (e.g., "123_two-sum")
    elif len(splits) == 2:
        return splits[0], splits[1]
    # Handle id only format
    else:
        return splits[0], None


def get_problem_key_from_data(data: dict) -> tuple[str | None, str, str]:
    """Extract source, problem_id, and title_slug from problem data.

    Args:
        data: The problem data dictionary

    Returns:
        A tuple of (source, problem_id, title_slug)
    """
    problem_id = str(data.get("id", 0)).replace("_", "").replace("-", "")
    title_slug = data.get("title_slug")
    source = data.get("source")

    if title_slug is None:
        title_slug = data.get("title", "").lower().replace(" ", "-").replace("_", "-").replace("/", "-")
    if not title_slug:
        title_slug = data.get("source", "unknown")

    return source, str(problem_id), title_slug


def pack_problems(
    input_dir: Path,
    output_file: Path,
    overwrite: bool,
    include: str = "*.json",
    exclude: str = None,
):
    """Pack individual problem JSON files into a single JSON file as a dictionary.

    Args:
        input_dir: Directory containing problem JSON files
        output_file: Path to the output JSON file
        overwrite: Whether to overwrite existing output file
        include: Glob pattern to include files
        exclude: Glob pattern to exclude files
    """
    if output_file.exists() and not overwrite:
        return

    files = input_dir.glob(include)
    if exclude:
        excluded_files = input_dir.glob(exclude)
        files = [f for f in files if f not in excluded_files]

    files = sort_problem_files(files)

    problems_dict = {}
    for file in tqdm(files, desc="Packing problems", total=len(files)):
        data = load_json_file(file)
        problems_dict[file.stem] = data

    save_json_file(output_file, problems_dict)


def unpack_problems(input_file: Path, output_dir: Path, overwrite: bool):
    """Unpack a dictionary-based JSON file into individual problem JSON files.

    Args:
        input_file: Path to the input JSON file
        output_dir: Directory to write individual JSON files
        overwrite: Whether to overwrite existing files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    problems_dict = load_json_file(input_file)

    for file_stem, data in tqdm(problems_dict.items(), desc="Unpacking problems", total=len(problems_dict)):
        # Use the original filename from the dictionary key
        output_file = output_dir / f"{file_stem}.json"

        if output_file.exists() and not overwrite:
            continue

        try:
            save_json_file(output_file, data)
        except Exception as e:
            logging.error(f"Error saving problem {output_file}: {e}")


def load_pack(pack_file: Path) -> dict:
    """Load packed JSON file as a dictionary of problems.

    Args:
        pack_file: Path to the packed JSON file

    Returns:
        Dictionary of problems with keys as original filenames
    """
    if not pack_file.exists():
        logging.error(f"Pack file {pack_file} does not exist")
        return {}

    # Load the JSON file directly
    return load_json_file(pack_file)


class MemoryExceededError(MemoryError):
    """Error raised when memory usage exceeds the specified threshold."""

    def __init__(self, current_mem_mb, delta_mem_mb, threshold_mb, context=None):
        """Initialize the memory exceeded error with usage details.

        Args:
            current_mem_mb: Current memory usage in MB
            delta_mem_mb: Increase in memory since monitoring started in MB
            threshold_mb: Memory threshold in MB
            context: Optional context string for the error message
        """
        self.current_mem_mb = current_mem_mb
        self.delta_mem_mb = delta_mem_mb
        self.threshold_mb = threshold_mb
        self.context = context

        message = f"Memory increased {delta_mem_mb:.1f}MB > threshold {threshold_mb}MB"
        if context:
            message = f"{context}: {message}"

        super().__init__(message)


class MemoryMonitor:
    """Utility class to monitor memory usage and enforce memory thresholds.

    This class provides a way to track memory usage throughout code execution
    and optionally raise an error when a specified threshold is exceeded.
    """

    def __init__(self, threshold_mb: int = 0):
        """Initialize the memory monitor.

        Args:
            threshold_mb: Memory threshold in MB. Set to 0 to disable threshold checking.
        """
        self.threshold_mb = threshold_mb
        self.start_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def check(self, context: str = ""):
        """Check current memory usage against the threshold.

        Args:
            context: Optional context string for error messages

        Returns:
            tuple: (current_mem_mb, delta_mem_mb) - Current memory and increase in MB

        Raises:
            MemoryExceededError: If memory usage exceeds the threshold
        """
        current_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        delta_mem_kb = current_mem_kb - self.start_mem_kb

        current_mem_mb = current_mem_kb / 1024
        delta_mem_mb = delta_mem_kb / 1024

        # Check threshold and raise error if exceeded
        if self.threshold_mb > 0 and delta_mem_mb > self.threshold_mb:
            raise MemoryExceededError(
                current_mem_mb=current_mem_mb,
                delta_mem_mb=delta_mem_mb,
                threshold_mb=self.threshold_mb,
                context=context,
            )

        return current_mem_mb, delta_mem_mb


def parse_distribution(
    dist_str: str | None,
    dist_name: str,
    lang: str,
    log,
) -> list[list[Any]] | None:
    """Parses a distribution string.

    Returns:
        The parsed list of [value, percentage, code_or_null], sorted by value.
        Returns None if critical parsing errors occur (e.g., JSON decode error, unexpected top-level structure).
        Returns an empty list if the distribution string is empty, or contains no valid items.
    """
    if not dist_str or not dist_str.strip():
        if dist_str is not None:  # Handles empty string case
            log(f"{dist_name} string is empty.", lang=lang, fg="yellow")
        return []  # Treat None or empty string as an empty distribution

    try:
        dist_data_raw = json.loads(dist_str)
    except json.JSONDecodeError:
        log(
            f"Failed to decode JSON for {dist_name}: '{dist_str[:100]}...'",
            lang=lang,
            fg="red",
        )
        return None  # Critical error, cannot parse

    if isinstance(dist_data_raw, list):
        is_origin_format = False
    elif (
        isinstance(dist_data_raw, dict)
        and "distribution" in dist_data_raw
        and isinstance(dist_data_raw["distribution"], list)
    ):
        dist_data_raw = dist_data_raw["distribution"]
        is_origin_format = True
    else:
        log(
            f"{dist_name} has an unexpected structure: '{dist_str[:100]}...'",
            lang=lang,
            fg="red",
        )
        return None  # Critical error, unrecognized structure

    dist_data: list[list[Any]] = []
    for i, item in enumerate(dist_data_raw):
        if not isinstance(item, list):
            log(
                f"{dist_name} item #{i} is not a list: {item}. Unexpected structure.",
                lang=lang,
                fg="yellow",
            )
            return None  # Critical error, unrecognized structure

        try:
            if is_origin_format:  # API format: ["value_str", percentage_float]
                if len(item) == 2:
                    dist_data.append([int(item[0]), float(item[1]), None])
                else:
                    log(
                        f"{dist_name} (from LeetCode) item #{i} has unexpected length {len(item)}: {item}. Unexpected structure.",
                        lang=lang,
                        fg="yellow",
                    )
                    return None  # Critical error, unrecognized structure
            else:  # Stored format: [value_int, percentage_float, code_or_null]
                if len(item) == 3:
                    dist_data.append(item)
                else:
                    log(
                        f"{dist_name} (from work file) item #{i} has unexpected length {len(item)}: {item}. Unexpected structure.",
                        lang=lang,
                        fg="yellow",
                    )
                    return None  # Critical error, unrecognized structure

        except (ValueError, TypeError) as e:
            log(
                f"{dist_name} item #{i} parsing error for '{item}': {e}. Unexpected structure.",
                lang=lang,
                fg="yellow",
            )
            return None  # Critical error, unrecognized structure

    return sorted(dist_data, key=lambda x: x[0])
