"""The results manifest: one JSON file with every reasoning result, its
checkpoint, the checkpoint's parent, its protocol and its source records.

  http.py         standard-library HTTP with retries, and range reads
  phase2.py       the three-seed Phase 2 records at a fixed dataset revision
  hf_snapshot.py  Hugging Face records for every checkpoint and its parents
  lineage.py      which checkpoint each checkpoint was trained from
  build.py        results, suite averages, changes, rejected pairings, checks
"""
