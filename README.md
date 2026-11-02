# permissive-post-train

The code that writes every table, figure and result number of *Strong
Post-Training from Permissive, Reasoning-Dominant Pretraining* (NeurIPS 2026) from its evaluation records, and checks them.

Every input is a record file pinned by its SHA-256 in
`mv_post_train/inputs.json`. They are all public: 29 in the Hugging Face
dataset
[`permissive-post-train/evaluation-records`](https://huggingface.co/datasets/permissive-post-train/evaluation-records),
and one small file in this repository, `config/paper_setup.toml`. `results/` holds every generated
file, as the paper prints it.

## Reproduce the paper's results

```bash
git clone https://github.com/MixtureVitae/permissive-post-train
cd permissive-post-train
pip install -r requirements-lock.txt   # the exact versions results/ was built with
pip install -e ".[test]"
mvpt fetch-inputs      # the 30 inputs of the paper, 9 GB, each checked by its SHA-256
mvpt build --check     # rebuild every generated file and compare it with results/
```

Python 3.12 (3.11 at least). `mvpt build --check` takes about 15 minutes and
3 GB of memory. It writes every file into a temporary folder and compares it
byte for byte with the committed one (a PNG preview by its pixels), and exits
1 if any differs. `mvpt build` writes them into `results/` instead. The
[reproduce workflow](.github/workflows/reproduce.yml) runs the same commands on
a clean GitHub runner.

## What it builds

| Folder | What it holds | Command |
|---|---|---|
| `results/manifest.json` | The results manifest: every checkpoint the paper evaluates, its parent, its Hugging Face repo, and every result with the record it comes from ([docs/manifest.md](docs/manifest.md)) | `mvpt build-manifest` |
| `results/zoo/` | Phase 1 for all models | `mvpt render-zoo` |
| `results/phase2/` | The Phase 2 tables and figures ([docs/phase2_tables.md](docs/phase2_tables.md)) | `mvpt render-phase2` |
| `results/phase2_analyses/` | The Phase 2 statistics: intervals, MixtureVitae against each reference, changes by domain, cut-off answers ([docs/phase2_analyses.md](docs/phase2_analyses.md)) | `mvpt render-phase2-analyses` |
| `results/frontloading/` | The front-loading study across four source pools ([docs/frontloading.md](docs/frontloading.md)) | `mvpt render-frontloading` |
| `results/older/` | The 100B mixture ablation, decontamination, the 0.4B comparison and masking ([docs/older_tables.md](docs/older_tables.md)) | `mvpt render-older-tables` |
| `results/language_suite/` | The language suite | `mvpt render-language-suite` |
| `results/language/` | Language changes with intervals ([docs/language_intervals.md](docs/language_intervals.md)) | `mvpt render-language-intervals` |
| `results/setup/` | The setup and evaluation-settings tables | `mvpt render-setup` |
| `results/statistics/` | The statistics appendix ([docs/statistics_appendix.md](docs/statistics_appendix.md)) | `mvpt render-statistics` |

The paper's folders hold LaTeX tables, figures, and a macro file that the
paper's text takes its numbers from. [docs/paper_build.md](docs/paper_build.md)
maps each file to its place in the paper's sources, and `mvpt check-paper
<paper sources>` compares a copy of the paper's sources with them.

Setup values that no evaluation record holds are kept in
`config/paper_setup.toml`, each with its source. Every result in `results/`
comes from an evaluation record.

## Inputs

[docs/inputs.md](docs/inputs.md) describes each input: what it holds, where
it comes from, and which tables read it. `mvpt fetch-inputs` downloads them,
`mvpt check-inputs` checks local copies, and `mv_post_train/inputs.json` lists
the sources of each one in the order they are tried. The records dataset's
card describes its files, and `release/records/` has the scripts that made and
uploaded them.

## License

The code is under the MIT license ([LICENSE](LICENSE)). The records in the
dataset are under CC BY 4.0, and the benchmark questions and model answers in
some of its files keep their own terms (see its card).
