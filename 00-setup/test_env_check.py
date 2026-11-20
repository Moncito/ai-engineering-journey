"""Tests for env_check.py — these are the spec. Do not edit; make them pass."""

import sys

import pytest
from env_check import mask_secret, parse_nvidia_smi, python_version


def test_python_version_matches_interpreter():
    v = sys.version_info
    assert python_version() == f"{v.major}.{v.minor}.{v.micro}"


def test_python_is_312_or_newer():
    assert sys.version_info >= (3, 12), "Run with `uv run`, not your system Python"


@pytest.mark.parametrize(
    ("secret", "expected"),
    [
        (None, "missing"),
        ("", "missing"),
        ("abc", "****"),
        ("12345678", "****"),
        ("123456789", "1234...6789"),
        ("sk-ant-api03-abcdefghijklmnop", "sk-a...mnop"),
    ],
)
def test_mask_secret(secret, expected):
    assert mask_secret(secret) == expected


def test_mask_secret_never_leaks_full_key():
    key = "sk-ant-api03-SUPERSECRETVALUE1234"
    assert key not in mask_secret(key)


def test_parse_single_gpu():
    out = "NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB\n"
    assert parse_nvidia_smi(out) == [
        {"name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU", "memory_mib": 4096}
    ]


def test_parse_multiple_gpus_and_blank_lines():
    out = "NVIDIA A100-SXM4-80GB, 81920 MiB\n\nNVIDIA A100-SXM4-80GB, 81920 MiB\n"
    assert parse_nvidia_smi(out) == [
        {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920},
        {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920},
    ]


def test_parse_empty_output():
    assert parse_nvidia_smi("") == []
