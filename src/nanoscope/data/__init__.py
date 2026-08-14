"""The data pipeline: turns a text corpus into memory-mapped token shards, plus a
seed-reproducible held-out split."""

from nanoscope.data.prepare import prepare

__all__ = ["prepare"]
