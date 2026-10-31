"""Model registry: directory name -> (family, phase, context_tier, tokens, size).

Hand-written. Directory names contain inconsistencies (long_sft_16k vs longsft_16k,
decontaminate vs decontaminated, ot30k_*_base with single underscore) so silent
regex matching would mis-attribute. Unmapped dirs raise.

Phases:
  base                base model (no post-training)
  sft                 Phase-1 Tulu3 SFT
  dpo                 Phase-1 Tulu3 DPO
  ot30k_on_sft        Phase-2 OpenThoughts3-30K applied to Phase-1 SFT
  ot30k_on_dpo        Phase-2 OpenThoughts3-30K applied to Phase-1 DPO
  ot30k_on_base       Phase-2 OT30K applied directly to long-SFT base (control)
  external_instruct   shipped instruct/DPO checkpoint from a third party

Families:
  MV         MixtureVitae (decontaminated 300B)
  MV-orig    MixtureVitae 300B v1 (non-decontaminated)
  MV-100     MixtureVitae 100B (full mixture, ablation reference)
  MV-noinst  MixtureVitae 100B without instruction subset
  MV-noweb   MixtureVitae 100B without web subset
  MV-merged  MixtureVitae merged variant
  MV-0.4B    MixtureVitae 0.4B
  Comma      CommonPile / Comma0.1
  Nemotron   Nemotron-CC-HQ (OpenSci-NM)
  FineWeb-Edu
  DCLM
  SmolLM2
  Qwen2.5
  Qwen3
  Baguettotron
  FW-Edu-ablation  ablation-model-fineweb-edu
"""
from __future__ import annotations

# (family, phase, context_tier, tokens_billion, size_billion)
REGISTRY: dict[str, tuple[str, str, str, float | None, float]] = {
    # External shipped checkpoints
    "HuggingFaceTB__SmolLM2-1.7B-Instruct": ("SmolLM2", "external_instruct", "8k", 11000, 1.7),
    "HuggingFaceTB__SmolLM2-1.7B-Instruct-16k": ("SmolLM2", "external_instruct", "16k", 11000, 1.7),
    "Qwen__Qwen2.5-1.5B": ("Qwen2.5", "base", "131k", 18000, 1.5),
    "Qwen__Qwen2.5-1.5B-Instruct": ("Qwen2.5", "external_instruct", "131k", 18000, 1.5),
    "Qwen__Qwen3-1.7B": ("Qwen3", "external_instruct", "32k", 36000, 1.7),
    "Qwen__Qwen3-1.7B-Base": ("Qwen3", "base", "32k", 36000, 1.7),
    "ontocord__SmolLM2-1.7B-16k": ("SmolLM2", "base", "16k", 11000, 1.7),
    # 0.4B MV variants
    "ali-elganzory__0.4b-mixturevitae-v1-decontaminated-300B-4096": ("MV-0.4B", "base", "4k", 300, 0.4),
    "ali-elganzory__0.4b-mixturevitae-v1-decontaminated-300B-4096-DPO-Tulu3-decontaminated": ("MV-0.4B", "dpo", "4k", 300, 0.4),
    "ali-elganzory__0.4b-mixturevitae-v1-decontaminated-300B-4096-SFT-Tulu3-decontaminated": ("MV-0.4B", "sft", "4k", 300, 0.4),
    "ali-elganzory__0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k": ("MV-0.4B", "base", "16k", 300, 0.4),
    "ali-elganzory__0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-DPO-Tulu3-decontaminated": ("MV-0.4B", "dpo", "16k", 300, 0.4),
    "ali-elganzory__0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-SFT-Tulu3-decontaminated": ("MV-0.4B", "sft", "16k", 300, 0.4),
    # Comma 1.7B 300B
    "ali-elganzory__1.7b-Comma0.1-300BT-WithChatTemplate": ("Comma", "base", "4k", 300, 1.7),
    "ali-elganzory__1.7b-Comma0.1-300BT-SFT-Tulu3-decontaminated": ("Comma", "sft", "4k", 300, 1.7),
    "ali-elganzory__1.7b-Comma0.1-300BT-DPO-Tulu3-decontaminated": ("Comma", "dpo", "4k", 300, 1.7),
    "ali-elganzory__1.7b-Comma0.1-300BT-longsft_16k": ("Comma", "base", "16k", 300, 1.7),
    "ali-elganzory__1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated": ("Comma", "sft", "16k", 300, 1.7),
    "ali-elganzory__1.7b-Comma0.1-300BT-longsft_16k-DPO-Tulu3-decontaminated": ("Comma", "dpo", "16k", 300, 1.7),
    # MV 100BT (ablation)
    "ali-elganzory__1.7b-MixtureVitae-100BT": ("MV-100", "base", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-100BT-SFT-Tulu3-decontaminated": ("MV-100", "sft", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-100BT-DPO-Tulu3-decontaminated": ("MV-100", "dpo", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-100BT-longsft_16k": ("MV-100", "base", "16k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-100BT-longsft_16k-SFT-Tulu3-decontaminated": ("MV-100", "sft", "16k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-100BT-longsft_16k-DPO-Tulu3-decontaminated": ("MV-100", "dpo", "16k", 100, 1.7),
    # MV-curated_instruct (no-web ablation)
    "ali-elganzory__1.7b-MixtureVitae-curated_instruct-100BT": ("MV-noweb", "base", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-curated_instruct-100BT-SFT-Tulu3-decontaminated": ("MV-noweb", "sft", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-curated_instruct-100BT-DPO-Tulu3-decontaminated": ("MV-noweb", "dpo", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k": ("MV-noweb", "base", "16k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-SFT-Tulu3-decontaminated": ("MV-noweb", "sft", "16k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-DPO-Tulu3-decontaminated": ("MV-noweb", "dpo", "16k", 100, 1.7),
    # MV-web_curated (no-instruct ablation)
    "ali-elganzory__1.7b-MixtureVitae-web_curated-100BT": ("MV-noinst", "base", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-web_curated-100BT-SFT-Tulu3-decontaminated": ("MV-noinst", "sft", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-web_curated-100BT-DPO-Tulu3-decontaminated": ("MV-noinst", "dpo", "4k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-web_curated-100BT-longsft_16k": ("MV-noinst", "base", "16k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-web_curated-100BT-longsft_16k-SFT-Tulu3-decontaminated": ("MV-noinst", "sft", "16k", 100, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-web_curated-100BT-longsft_16k-DPO-Tulu3-decontaminated": ("MV-noinst", "dpo", "16k", 100, 1.7),
    # MV 300BT v1 non-decontaminated
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-WithChatTemplate": ("MV-orig", "base", "4k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-SFT-Tulu3": ("MV-orig", "sft", "4k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-DPO-Tulu3": ("MV-orig", "dpo", "4k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-16k-WithChatTemplate": ("MV-orig", "base", "16k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-16k-SFT-Tulu3": ("MV-orig", "sft", "16k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-16k-DPO-Tulu3": ("MV-orig", "dpo", "16k", 300, 1.7),
    # MV 300BT v1 decontaminated (the headline MV)
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated": ("MV", "base", "4k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-SFT-Tulu3-decontaminated": ("MV", "sft", "4k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-DPO-Tulu3-decontaminated": ("MV", "dpo", "4k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-16k": ("MV", "base", "16k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated": ("MV", "sft", "16k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated": ("MV", "dpo", "16k", 300, 1.7),
    # MV 300BT v1 decontaminated merged (alternative checkpoint)
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged": ("MV-merged", "base", "16k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-SFT-Tulu3-decontaminated": ("MV-merged", "sft", "16k", 300, 1.7),
    "ali-elganzory__1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-DPO-Tulu3-decontaminated": ("MV-merged", "dpo", "16k", 300, 1.7),
    # Baguettotron
    "ali-elganzory__Baguettotron": ("Baguettotron", "base", "4k", None, 0.32),
    "ali-elganzory__Baguettotron-SFT-Tulu3-decontaminated": ("Baguettotron", "sft", "4k", None, 0.32),
    "ali-elganzory__Baguettotron-DPO-Tulu3-decontaminated": ("Baguettotron", "dpo", "4k", None, 0.32),
    "ali-elganzory__Baguettotron-longsft_16k": ("Baguettotron", "base", "16k", None, 0.32),
    "ali-elganzory__Baguettotron-longsft_16k-SFT-Tulu3-decontaminated": ("Baguettotron", "sft", "16k", None, 0.32),
    "ali-elganzory__Baguettotron-longsft_16k-DPO-Tulu3-decontaminated": ("Baguettotron", "dpo", "16k", None, 0.32),
    # Qwen / SmolLM Tulu3-applied (controlled comparison vs released instruct)
    "ali-elganzory__Qwen2.5-1.5B-SFT-Tulu3-decontaminated": ("Qwen2.5", "sft", "131k", 18000, 1.5),
    "ali-elganzory__Qwen2.5-1.5B-DPO-Tulu3-decontaminated": ("Qwen2.5", "dpo", "131k", 18000, 1.5),
    "ali-elganzory__Qwen3-1.7B-Base-SFT-Tulu3-decontaminated": ("Qwen3", "sft", "32k", 36000, 1.7),
    "ali-elganzory__Qwen3-1.7B-Base-DPO-Tulu3-decontaminated": ("Qwen3", "dpo", "32k", 36000, 1.7),
    "ali-elganzory__SmolLM2-1.7B-WithChatTemplate": ("SmolLM2", "base", "8k", 11000, 1.7),
    "ali-elganzory__SmolLM2-1.7B-SFT-Tulu3-decontaminated": ("SmolLM2", "sft", "8k", 11000, 1.7),
    "ali-elganzory__SmolLM2-1.7B-DPO-Tulu3-decontaminated": ("SmolLM2", "dpo", "8k", 11000, 1.7),
    "ali-elganzory__SmolLM2-1.7B-16k": ("SmolLM2", "base", "16k", 11000, 1.7),
    "ali-elganzory__SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated": ("SmolLM2", "sft", "16k", 11000, 1.7),
    "ali-elganzory__SmolLM2-1.7B-16k-DPO-Tulu3-decontaminated": ("SmolLM2", "dpo", "16k", 11000, 1.7),
    # Ablation FineWeb-Edu (older "ablation-model-fineweb-edu")
    "ali-elganzory__ablation-model-fineweb-edu-WithChatTemplate": ("FW-Edu-ablation", "base", "4k", None, 1.7),
    "ali-elganzory__ablation-model-fineweb-edu-SFT-Tulu3-decontaminated": ("FW-Edu-ablation", "sft", "4k", None, 1.7),
    "ali-elganzory__ablation-model-fineweb-edu-DPO-Tulu3-decontaminated": ("FW-Edu-ablation", "dpo", "4k", None, 1.7),
    # Nemotron v0.01 (16384 ctx, older recipe)
    "ali-elganzory__open-sci-ref-v0.01-1.7b-nemotron-hq-300B-16384-WithChatTemplate": ("Nemotron-v01", "base", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.01-1.7b-nemotron-hq-300B-16384-SFT-Tulu3": ("Nemotron-v01", "sft", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.01-1.7b-nemotron-hq-300B-16384-DPO-Tulu3": ("Nemotron-v01", "dpo", "16k", 300, 1.7),
    # DCLM v0.02
    "ali-elganzory__open-sci-ref-v0.02-1.7b-dclm-300B-4096": ("DCLM", "base", "4k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-dclm-300B-4096-SFT-Tulu3-decontaminated": ("DCLM", "sft", "4k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-dclm-300B-4096-DPO-Tulu3-decontaminated": ("DCLM", "dpo", "4k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k": ("DCLM", "base", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-SFT-Tulu3-decontaminated": ("DCLM", "sft", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-DPO-Tulu3-decontaminated": ("DCLM", "dpo", "16k", 300, 1.7),
    # FineWeb-Edu v0.02
    "ali-elganzory__open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096": ("FineWeb-Edu", "base", "4k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-SFT-Tulu3-decontaminated": ("FineWeb-Edu", "sft", "4k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-DPO-Tulu3-decontaminated": ("FineWeb-Edu", "dpo", "4k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k": ("FineWeb-Edu", "base", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-Tulu3-decontaminated": ("FineWeb-Edu", "sft", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-DPO-Tulu3-decontaminated": ("FineWeb-Edu", "dpo", "16k", 300, 1.7),
    # Base checkpoints uploaded under non-`ali-elganzory__` accounts. Only the
    # language-eval logs zip references these directories (the reasoning
    # logs.zip uses the `ali-elganzory__` mirrors above).
    "ontocord__1.7b-Comma0.1-300BT": ("Comma", "base", "4k", 300, 1.7),
    "ontocord__1.7b-Comma0.1-300BT-longsft_16k": ("Comma", "base", "16k", 300, 1.7),
    "ontocord__1.7b-MixtureVitae-300BT-v1-decontaminated": ("MV", "base", "4k", 300, 1.7),
    "ontocord__1.7b-MixtureVitae-300BT-v1-decontaminated-16k": ("MV", "base", "16k", 300, 1.7),
    "open-sci__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096": ("Nemotron", "base", "4k", 300, 1.7),
    "open-sci__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k": ("Nemotron", "base", "16k", 300, 1.7),
    # FineWeb-Edu language-eval base checkpoints. The 16k base appears under
    # two different (broken) naming conventions in the supplied zip — both
    # carry dots replaced with underscores in some token positions (`1_7b`,
    # `v0_02`, `1_4t`). They're the same checkpoint logically; both map to
    # the FineWeb-Edu base 16k cell.
    "open-sci__open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096": ("FineWeb-Edu", "base", "4k", 300, 1.7),
    "open-sci__open-sci-ref-v0.02-1_7b-fineweb-edu-1.4t-300B-4096-longsft_16k": ("FineWeb-Edu", "base", "16k", 300, 1.7),
    "open-sci__open-sci-ref-v0_02-1_7b-fineweb-edu-1_4t-300B-4096-longsft_16k": ("FineWeb-Edu", "base", "16k", 300, 1.7),
    # Nemotron v0.02 4K
    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096": ("Nemotron", "base", "4k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-SFT-Tulu3-decontaminated": ("Nemotron", "sft", "4k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-DPO-Tulu3-decontaminated": ("Nemotron", "dpo", "4k", 300, 1.7),
    # Nemotron v0.02 16K. The paper's Table 1 row "OpenSci-NM-16K" matches the
    # `-300B-16k-` named variant; `-300B-4096-long_sft_16k-` is a separate
    # run (different HumanEval / MBPP scores). We treat the `-4096-` one as a
    # parallel "Nemotron-altpath" base/sft so the canonical Nemotron family
    # has exactly one model per (phase, ctx) cell.
    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k": ("Nemotron-altpath", "base", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k-SFT-Tulu3-decontaminated": ("Nemotron-altpath", "sft", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16384-rope_theta-1M-long_sft_16k": ("Nemotron-rope1M", "base", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-SFT-Tulu3-decontaminated": ("Nemotron", "sft", "16k", 300, 1.7),
    "ali-elganzory__open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-DPO-Tulu3-decontaminated": ("Nemotron", "dpo", "16k", 300, 1.7),
    # OT30K Phase 2 (note: 'decontaminate' vs 'decontaminated' truncation in Comma DPO)
    "open-sci__sft__ot30k_1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated": ("Comma", "ot30k_on_sft", "16k", 300, 1.7),
    "open-sci__sft__ot30k_1.7b-Comma0.1-300BT-longsft_16k-DPO-Tulu3-decontaminate": ("Comma", "ot30k_on_dpo", "16k", 300, 1.7),
    "open-sci__sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k": ("MV", "ot30k_on_base", "16k", 300, 1.7),
    "open-sci__sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated": ("MV", "ot30k_on_sft", "16k", 300, 1.7),
    "open-sci__sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated": ("MV", "ot30k_on_dpo", "16k", 300, 1.7),
    "open-sci__sft__ot30k_Qwen2.5-1.5B-SFT-Tulu3-decontaminated": ("Qwen2.5", "ot30k_on_sft", "131k", 18000, 1.5),
    "open-sci__sft__ot30k_Qwen2.5-1.5B-DPO-Tulu3-decontaminated": ("Qwen2.5", "ot30k_on_dpo", "131k", 18000, 1.5),
    "open-sci__sft__ot30k_Qwen3-1.7B-Base-SFT-Tulu3-decontaminated": ("Qwen3", "ot30k_on_sft", "32k", 36000, 1.7),
    "open-sci__sft__ot30k_Qwen3-1.7B-Base-DPO-Tulu3-decontaminated": ("Qwen3", "ot30k_on_dpo", "32k", 36000, 1.7),
    "open-sci__sft__ot30k_SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated": ("SmolLM2", "ot30k_on_sft", "16k", 11000, 1.7),
    "open-sci__sft__ot30k_SmolLM2-1.7B-Instruct-16k": ("SmolLM2", "ot30k_on_base", "16k", 11000, 1.7),
    "open-sci__sft__ot30k_open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-Tulu3": ("FineWeb-Edu", "ot30k_on_sft", "16k", 300, 1.7),
    "open-sci__sft__ot30k_open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-SFT-Tulu3-decontaminated": ("Nemotron", "ot30k_on_sft", "16k", 300, 1.7),
    "open-sci__sft__ot30k_open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-DPO-Tulu3-decontaminated": ("Nemotron", "ot30k_on_dpo", "16k", 300, 1.7),
    # OT30K applied directly to MV-16K base — alternate run, distinct from the canonical
    # `open-sci__sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k` above (different
    # numbers on HumanEval/MATH500/IFEval). Kept as a separate phase so they don't conflate.
    "open-sci__sft_ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k_base": ("MV", "ot30k_on_base_alt", "16k", 300, 1.7),
}


# ---------------------------------------------------------------------------
# Phase taxonomy. The paper splits training into two phases:
#   Phase 1: base pretraining + Tulu3 SFT/DPO post-training.
#   Phase 2: OpenThoughts3-30K reasoning post-training applied to a Phase 1
#            checkpoint (or directly to the base).
#
# Clean source separation: logs.zip emits Phase 1 (and reference / external
# checkpoints) only; the Phase 2 HF parquet emits Phase 2 only.
# `data/parse_phase1.py` filters out PHASE2_PHASES; `data/parse_phase2.py`
# emits only those phases.
PHASE2_PHASES: frozenset[str] = frozenset({
    "ot30k_on_base",
    "ot30k_on_base_alt",
    "ot30k_on_sft",
    "ot30k_on_dpo",
})


def lookup(directory_name: str) -> tuple[str, str, str, float | None, float]:
    if directory_name not in REGISTRY:
        raise KeyError(f"Unmapped model directory: {directory_name}")
    return REGISTRY[directory_name]


def known_dirs() -> set[str]:
    return set(REGISTRY.keys())
