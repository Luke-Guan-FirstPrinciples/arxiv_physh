#!/usr/bin/env python3
"""
Proof-of-concept physics-focused keyword extraction from title + abstract.

Pipeline:
1) Build text from title + abstract
2) Generate candidate keywords with KeyBERT using SciBERT embeddings
3) Re-rank candidates with a lightweight physics-specific score boost
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Sequence, Tuple

import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


PHYSICS_TOKEN_HINTS = {
    "adiabatic",
    "astrophysics",
    "atom",
    "atomic",
    "boson",
    "bosonic",
    "cosmic",
    "cosmology",
    "dark",
    "diffusion",
    "electrodynamics",
    "electromagnetic",
    "electron",
    "entropy",
    "fermion",
    "fermionic",
    "fluid",
    "gauge",
    "gravity",
    "gravitational",
    "hadron",
    "hamiltonian",
    "heisenberg",
    "higgs",
    "holographic",
    "inflation",
    "ising",
    "lattice",
    "lagrangian",
    "magnetization",
    "magnetic",
    "many-body",
    "meson",
    "molecular",
    "neutrino",
    "neutron",
    "nuclear",
    "order-parameter",
    "oscillation",
    "particle",
    "perturbation",
    "phase",
    "phonon",
    "photon",
    "plasma",
    "polymer",
    "quantization",
    "quantum",
    "quark",
    "relativity",
    "renormalization",
    "scattering",
    "spin",
    "spectroscopy",
    "spacetime",
    "superconducting",
    "superconductivity",
    "symmetry",
    "thermodynamic",
    "thermodynamics",
    "topological",
    "vortex",
    "wavefunction",
}

PHYSICS_PHRASE_HINTS = {
    "critical exponents",
    "dark matter",
    "density functional theory",
    "effective field theory",
    "equation of state",
    "feynman diagram",
    "gauge field",
    "ground state",
    "hilbert space",
    "landau level",
    "matrix element",
    "mean field",
    "phase transition",
    "quantum hall",
    "quantum monte carlo",
    "renormalization group",
    "schrodinger equation",
    "spin orbit coupling",
    "standard model",
    "vacuum expectation value",
}

GENERIC_PENALTY_TOKENS = {
    "analysis",
    "approach",
    "data",
    "dataset",
    "framework",
    "method",
    "methods",
    "model",
    "result",
    "results",
    "study",
    "system",
}


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def combine_title_abstract(title: str | None, abstract: str | None) -> str:
    title_clean = clean_text(title)
    abstract_clean = clean_text(abstract)
    if title_clean and abstract_clean:
        return f"{title_clean} [SEP] {abstract_clean}"
    if title_clean:
        return title_clean
    return abstract_clean


def tokenize_lower(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def resolve_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def physics_boost(keyword: str, title_tokens: set[str]) -> float:
    tokens = tokenize_lower(keyword)
    if not tokens:
        return 0.0

    boost = 0.0
    matched_tokens = sum(1 for token in tokens if token in PHYSICS_TOKEN_HINTS)
    if matched_tokens:
        boost += min(0.22, 0.06 * matched_tokens)

    if any(phrase in keyword for phrase in PHYSICS_PHRASE_HINTS):
        boost += 0.12

    title_overlap = len(set(tokens) & title_tokens)
    if title_overlap:
        boost += min(0.12, 0.04 * title_overlap)

    if len(tokens) == 1 and tokens[0] in GENERIC_PENALTY_TOKENS:
        boost -= 0.08

    return max(-0.1, min(0.4, boost))


def extract_physics_keywords(
    title: str,
    abstract: str,
    model_name: str,
    device: str,
    ngram_range: Tuple[int, int],
    candidate_top_n: int,
    top_n: int,
    diversity: float,
) -> List[Dict[str, float | str]]:
    text = combine_title_abstract(title, abstract)
    if not text:
        raise ValueError("Provide at least one of --title or --abstract.")

    encoder = SentenceTransformer(model_name, device=device)
    kw_model = KeyBERT(model=encoder)

    raw_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words="english",
        top_n=candidate_top_n,
        use_mmr=diversity > 0,
        diversity=max(0.0, min(1.0, diversity)),
    )

    title_tokens = set(tokenize_lower(title))
    rescored: List[Dict[str, float | str]] = []
    seen: set[str] = set()

    for keyword, base_score in raw_keywords:
        normalized_keyword = clean_text(keyword).lower()
        if not normalized_keyword or normalized_keyword in seen:
            continue
        seen.add(normalized_keyword)

        boost = physics_boost(normalized_keyword, title_tokens)
        final_score = float(base_score) + boost
        rescored.append(
            {
                "keyword": normalized_keyword,
                "score": round(final_score, 6),
                "base_score": round(float(base_score), 6),
                "physics_boost": round(boost, 6),
            }
        )

    rescored.sort(key=lambda row: (row["score"], row["base_score"]), reverse=True)
    return rescored[:top_n]


def print_keywords_table(keywords: Sequence[Dict[str, float | str]]) -> None:
    print(f"{'rank':<4} {'keyword':<45} {'score':>8} {'base':>8} {'boost':>8}")
    print("-" * 80)
    for idx, row in enumerate(keywords, start=1):
        print(
            f"{idx:<4} "
            f"{str(row['keyword'])[:45]:<45} "
            f"{float(row['score']):>8.4f} "
            f"{float(row['base_score']):>8.4f} "
            f"{float(row['physics_boost']):>8.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract physics-focused keywords from title + abstract."
    )
    parser.add_argument("--title", type=str, default="", help="Paper title text")
    parser.add_argument("--abstract", type=str, default="", help="Paper abstract text")
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Embedding model for KeyBERT",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (auto picks cuda/mps/cpu)",
    )
    parser.add_argument(
        "--min-ngram",
        type=int,
        default=1,
        help="Minimum keyphrase n-gram size",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        default=3,
        help="Maximum keyphrase n-gram size",
    )
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=30,
        help="Candidates from KeyBERT before physics re-ranking",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Final number of returned keywords",
    )
    parser.add_argument(
        "--diversity",
        type=float,
        default=0.35,
        help="MMR diversity in [0,1]. Set to 0 to disable MMR.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of a table",
    )
    args = parser.parse_args()

    if args.min_ngram < 1 or args.max_ngram < 1:
        parser.error("--min-ngram and --max-ngram must be >= 1")
    if args.min_ngram > args.max_ngram:
        parser.error("--min-ngram cannot be larger than --max-ngram")
    if args.top_n < 1 or args.candidate_top_n < 1:
        parser.error("--top-n and --candidate-top-n must be >= 1")

    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    keywords = extract_physics_keywords(
        title=args.title,
        abstract=args.abstract,
        model_name=args.model,
        device=device,
        ngram_range=(args.min_ngram, args.max_ngram),
        candidate_top_n=args.candidate_top_n,
        top_n=args.top_n,
        diversity=args.diversity,
    )

    print(f"# model={args.model} device={device}")
    if args.json:
        print(json.dumps(keywords, indent=2))
    else:
        print_keywords_table(keywords)


if __name__ == "__main__":
    main()
