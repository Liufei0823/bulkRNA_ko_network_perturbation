#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bulk_ko_auto_symbol_ensembl.py

Small-sample bulk RNA-seq prior-network virtual knockout impact ranking with automatic gene-symbol / Ensembl ID handling.

What this script does
---------------------
1) Reads a control bulk RNA expression matrix (genes x samples).
2) Downloads or reads a prior gene network.
   - Default automatic prior network: OmniPath signed directed interactions.
   - You may also provide your own source-target network.
3) Filters and optionally expression-weights the network using the control RNA expression.
4) For each candidate KO gene, sets a negative seed perturbation and propagates it
   through the signed network.
5) Outputs KO-specific ranked genes by perturbation impact score.
6) Optionally calls Enrichr online for GO/KEGG/Reactome/Hallmark/WikiPathways/BioPlanet/HumanCyc.
7) Optionally performs offline GMT enrichment if a GMT file is supplied.

This is NOT a transcriptome generator. It does NOT output true post-KO expression.
It outputs signed prior-network propagation scores, which are exploratory impact
rankings and should not be interpreted as measured or predicted log fold changes.

Dependencies
------------
pip install pandas numpy scipy requests openpyxl xlrd
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.stats import hypergeom


# -----------------------------
# General utilities
# -----------------------------

def mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        running = min(running, val)
        adj[i] = running
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out


def normalize_gene_symbol(x: object) -> str:
    return str(x).strip().upper()


def parse_bool(x: object) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "stimulates", "stimulation", "positive"}


def safe_float(x: object, default: float = 1.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Gene / protein identifier utilities
# -----------------------------

UNIPROT_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])(?:-\d+)?$")
ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G[0-9]+(?:\.\d+)?$", re.IGNORECASE)


def strip_ensembl_version(x: object) -> str:
    s = str(x).strip().upper()
    if "." in s and ENSEMBL_RE.match(s):
        s = s.split(".", 1)[0]
    return s


def looks_uniprot_id(x: object) -> bool:
    s = str(x).strip().upper()
    return bool(UNIPROT_RE.match(s))


def looks_ensembl_gene_id(x: object) -> bool:
    return bool(ENSEMBL_RE.match(str(x).strip().upper()))


def infer_id_type(values: Sequence[object], sample_n: int = 200) -> str:
    vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().lower() != "nan"][:sample_n]
    if not vals:
        return "symbol"
    frac_ensg = np.mean([looks_ensembl_gene_id(v) for v in vals])
    frac_uniprot = np.mean([looks_uniprot_id(v) for v in vals])
    if frac_ensg >= 0.5:
        return "ensembl"
    if frac_uniprot >= 0.5:
        return "uniprot"
    return "symbol"


def normalize_for_id_type(x: object, id_type: str) -> str:
    if id_type == "ensembl":
        return strip_ensembl_version(x)
    # symbol and uniprot both use upper-case tokens for matching in this script.
    return normalize_gene_symbol(x)


def read_gene_map(gene_map_path: Optional[str | Path]) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Read an optional gene identifier map.

    Supported formats:
      1) Headered table with any of these columns:
         symbol / gene_symbol / hgnc_symbol / external_gene_name / gene_name
         ensembl / ensg / ensembl_gene_id / gene_id / ensembl_id
         uniprot / uniprot_id / uniprotkb / accession / protein_id

      2) Headerless 2-column table, like:
         ENSG00000000003    TSPAN6
         ENSG00000000005    TNMD
         In this case the script auto-detects which column is Ensembl and which is symbol.

    The returned mapping is used to convert Ensembl expression IDs or KO IDs to
    gene symbols, because the recommended OmniPath network columns are
    source_genesymbol and target_genesymbol.
    """
    if not gene_map_path:
        return None

    path = Path(gene_map_path)
    # Try to decide if the file has a header. The user-provided gene.tsv example
    # has no header: ENSG...<tab>SYMBOL.
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline().strip('\n')
    first_parts = re.split(r'\t|,', first)
    first_parts = [x.strip() for x in first_parts if x.strip()]
    first_has_header = any(x.lower() in {
        'symbol', 'gene_symbol', 'hgnc_symbol', 'external_gene_name', 'gene_name',
        'ensembl', 'ensg', 'ensembl_gene_id', 'gene_id', 'ensembl_id',
        'uniprot', 'uniprot_id', 'uniprotkb', 'uniprotkb_id', 'accession', 'protein_id'
    } for x in first_parts)

    if first_has_header:
        df = pd.read_csv(path, sep=None, engine='python')
    else:
        df = pd.read_csv(path, sep=None, engine='python', header=None)
        if df.shape[1] < 2:
            raise ValueError('--gene_map without a header must have at least two columns: Ensembl and symbol.')
        # Auto-detect Ensembl and symbol columns from the first two/three columns.
        cols = list(df.columns)
        ensembl_col = None
        symbol_col = None
        uniprot_col = None
        for c in cols:
            vals = df[c].dropna().astype(str).head(200).tolist()
            typ = infer_id_type(vals)
            if typ == 'ensembl' and ensembl_col is None:
                ensembl_col = c
            elif typ == 'uniprot' and uniprot_col is None:
                uniprot_col = c
            elif typ == 'symbol' and symbol_col is None:
                symbol_col = c
        if ensembl_col is None or symbol_col is None:
            # Fallback for the common two-column format: ENSG, SYMBOL.
            ensembl_col = cols[0]
            symbol_col = cols[1]
        rename = {ensembl_col: 'ensembl', symbol_col: 'symbol'}
        if uniprot_col is not None:
            rename[uniprot_col] = 'uniprot'
        df = df.rename(columns=rename)

    lower = {str(c).lower().strip(): c for c in df.columns}

    def col(cands: Sequence[str]) -> Optional[str]:
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    symbol_col = col(['symbol', 'gene_symbol', 'hgnc_symbol', 'external_gene_name', 'gene_name'])
    ensembl_col = col(['ensembl', 'ensg', 'ensembl_gene_id', 'gene_id', 'ensembl_id'])
    uniprot_col = col(['uniprot', 'uniprot_id', 'uniprotkb', 'uniprotkb_id', 'accession', 'protein_id'])

    if symbol_col is None and ensembl_col is None and uniprot_col is None:
        raise ValueError('Could not identify any usable columns in --gene_map. Use symbol/ensembl/uniprot columns or a 2-column ENSG<TAB>SYMBOL file.')

    maps: Dict[str, Dict[str, str]] = {
        'ensembl_to_symbol': {},
        'symbol_to_ensembl': {},
        'uniprot_to_symbol': {},
        'symbol_to_uniprot': {},
        'uniprot_to_ensembl': {},
        'ensembl_to_uniprot': {},
    }

    for _, row in df.iterrows():
        symbol = normalize_gene_symbol(row[symbol_col]) if symbol_col is not None else ''
        ensembl = strip_ensembl_version(row[ensembl_col]) if ensembl_col is not None else ''
        uniprot = normalize_gene_symbol(row[uniprot_col]) if uniprot_col is not None else ''
        if symbol.lower() == 'nan':
            symbol = ''
        if ensembl.lower() == 'nan':
            ensembl = ''
        if uniprot.lower() == 'nan':
            uniprot = ''

        if symbol and ensembl:
            maps['ensembl_to_symbol'].setdefault(ensembl, symbol)
            maps['symbol_to_ensembl'].setdefault(symbol, ensembl)
        if symbol and uniprot:
            maps['uniprot_to_symbol'].setdefault(uniprot, symbol)
            maps['symbol_to_uniprot'].setdefault(symbol, uniprot)
        if ensembl and uniprot:
            maps['uniprot_to_ensembl'].setdefault(uniprot, ensembl)
            maps['ensembl_to_uniprot'].setdefault(ensembl, uniprot)

    return maps


def convert_identifier(x: object, from_type: str, to_type: str, gene_map: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    if from_type == "auto":
        from_type = infer_id_type([x])
    if from_type == to_type:
        return normalize_for_id_type(x, to_type)
    raw = normalize_for_id_type(x, from_type)
    if not gene_map:
        return raw
    key = f"{from_type}_to_{to_type}"
    if key in gene_map:
        return gene_map[key].get(raw, "")
    return raw


def choose_identifier_column(
    net: pd.DataFrame,
    kind: str,
    analysis_id_type: str,
    network_node_id: str = "auto",
) -> Optional[str]:
    lower = {c.lower().strip(): c for c in net.columns}

    def find(cands: Sequence[str]) -> Optional[str]:
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    if analysis_id_type == "symbol" and network_node_id in {"auto", "symbol"}:
        c = find([f"{kind}_genesymbol", f"{kind}_gene_symbol", f"{kind}_symbol", f"{kind}_label"])
        if c is not None:
            return c
    if analysis_id_type == "ensembl" and network_node_id in {"auto", "ensembl"}:
        c = find([f"{kind}_ensembl", f"{kind}_ensg", f"{kind}_ensembl_gene_id"])
        if c is not None:
            return c
    return find([kind, f"{kind}_gene", "from" if kind == "source" else "to"])


def map_network_node(
    x: object,
    source_id_type: str,
    analysis_id_type: str,
    gene_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    # Complexes do not correspond to a single RNA-seq gene row.
    if s.upper().startswith("COMPLEX:"):
        return ""
    if source_id_type == "auto":
        source_id_type = infer_id_type([s])
    if source_id_type == analysis_id_type:
        return normalize_for_id_type(s, analysis_id_type)
    mapped = convert_identifier(s, source_id_type, analysis_id_type, gene_map=gene_map)
    return mapped


# -----------------------------
# Input expression
# -----------------------------

def read_expression(
    expr_path: str | Path,
    gene_col: str = "gene",
    expr_id_type: str = "auto",
    analysis_id_type: str = "symbol",
    gene_map: Optional[Dict[str, Dict[str, str]]] = None,
    unmapped_out: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Read gene x sample expression matrix and normalize gene IDs.

    The analysis is normally performed in gene SYMBOL space because OmniPath can
    provide gene symbols. Expression may be SYMBOL or Ensembl ID. If expression
    uses Ensembl IDs and analysis_id_type is symbol, provide --gene_map.
    """
    expr_path = Path(expr_path)
    if expr_path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        try:
            expr = pd.read_excel(expr_path)
        except Exception as exc:
            # Some GEO/R exports use an Excel-looking suffix for tabular text.
            try:
                expr = pd.read_csv(expr_path, sep=None, engine="python")
            except Exception:
                raise ValueError(f"Could not read expression file as Excel or delimited text: {expr_path}") from exc
    else:
        expr = pd.read_csv(expr_path, sep=None, engine="python")
    if gene_col not in expr.columns:
        first = expr.columns[0]
        expr = expr.rename(columns={first: gene_col})

    raw_ids = expr[gene_col].astype(str).tolist()
    observed_type = infer_id_type(raw_ids) if expr_id_type == "auto" else expr_id_type

    mapped_ids = []
    unmapped = []
    for g in raw_ids:
        if observed_type == analysis_id_type:
            mg = normalize_for_id_type(g, analysis_id_type)
        else:
            mg = convert_identifier(g, observed_type, analysis_id_type, gene_map=gene_map)
        if not mg:
            unmapped.append(g)
        mapped_ids.append(mg)

    if unmapped_out is not None and unmapped:
        pd.DataFrame({"unmapped_expression_gene_id": sorted(set(unmapped))}).to_csv(unmapped_out, sep="\t", index=False)

    expr[gene_col] = mapped_ids
    expr = expr[(expr[gene_col] != "") & (expr[gene_col].astype(str).str.lower() != "nan")].copy()
    sample_cols = [c for c in expr.columns if c != gene_col]
    if len(sample_cols) == 0:
        raise ValueError("Expression file must contain at least one sample column.")

    for c in sample_cols:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")
    if expr[sample_cols].isna().all(axis=None):
        raise ValueError("All expression values are NA after numeric conversion.")

    expr = expr.groupby(gene_col, as_index=True)[sample_cols].mean()
    expr.index.name = "gene"
    return expr


def read_ko_genes(
    ko_path: str | Path,
    ko_id_type: str = "auto",
    analysis_id_type: str = "symbol",
    gene_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[List[str], List[str]]:
    raw = []
    with open(ko_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # allow one-column text or first field of TSV/CSV-like lines
            s = re.split(r"[\t,]", s)[0].strip()
            if s and s.lower() not in {"gene", "geneid", "ko", "ko_gene"}:
                raw.append(s)
    raw = list(dict.fromkeys(raw))
    observed_type = infer_id_type(raw) if ko_id_type == "auto" else ko_id_type
    mapped = []
    missing = []
    for g in raw:
        if observed_type == analysis_id_type:
            mg = normalize_for_id_type(g, analysis_id_type)
        else:
            mg = convert_identifier(g, observed_type, analysis_id_type, gene_map=gene_map)
        if mg:
            mapped.append(mg)
        else:
            missing.append(g)
    return list(dict.fromkeys(mapped)), missing

def compute_gene_stats(expr: pd.DataFrame, detected_threshold: float = 0.0) -> pd.DataFrame:
    stats = pd.DataFrame(index=expr.index)
    stats["mean_expr"] = expr.mean(axis=1)
    stats["median_expr"] = expr.median(axis=1)
    stats["sd_expr"] = expr.std(axis=1, ddof=1).fillna(0.0)
    stats["detected_fraction"] = (expr > detected_threshold).mean(axis=1)
    return stats


# -----------------------------
# Network loading and downloading
# -----------------------------

def standardize_network_columns(
    net: pd.DataFrame,
    network_node_id: str = "auto",
    analysis_id_type: str = "symbol",
    gene_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Standardize a prior network to source, target, weight, sign.

    Important: OmniPath /interactions may return UniProt accessions in source/target.
    This function prefers source_genesymbol/target_genesymbol when present; otherwise
    it can convert UniProt or Ensembl nodes with --gene_map. Complex nodes are dropped.
    """
    if net.empty:
        raise ValueError("Network table is empty.")

    colmap = {c.lower().strip(): c for c in net.columns}

    def find_col(candidates: Sequence[str]) -> Optional[str]:
        for cand in candidates:
            if cand.lower() in colmap:
                return colmap[cand.lower()]
        return None

    source_col = choose_identifier_column(net, "source", analysis_id_type, network_node_id)
    target_col = choose_identifier_column(net, "target", analysis_id_type, network_node_id)

    if source_col is None or target_col is None:
        raise ValueError(
            "Network must contain source/target columns or source/target gene symbol columns. "
            f"Observed columns: {list(net.columns)}"
        )

    # Determine the identifier type stored in the selected columns.
    if network_node_id == "auto":
        if source_col.lower().endswith("genesymbol") or "symbol" in source_col.lower():
            src_type = tgt_type = "symbol"
        elif "ensembl" in source_col.lower() or "ensg" in source_col.lower():
            src_type = tgt_type = "ensembl"
        else:
            src_type = infer_id_type(net[source_col].dropna().astype(str).head(300).tolist())
            tgt_type = infer_id_type(net[target_col].dropna().astype(str).head(300).tolist())
    else:
        src_type = tgt_type = network_node_id

    # If the selected columns are UniProt and no map is supplied, the network will not
    # match symbol/Ensembl expression. Fail early with a clear message.
    if analysis_id_type in {"symbol", "ensembl"} and (src_type == "uniprot" or tgt_type == "uniprot") and gene_map is None:
        raise ValueError(
            "The network source/target columns appear to contain UniProt protein IDs, "
            f"but analysis_id_type={analysis_id_type!r}. Provide --gene_map with columns "
            "symbol, ensembl and/or uniprot, or download an OmniPath table that includes "
            "source_genesymbol and target_genesymbol columns."
        )

    out = pd.DataFrame()
    out["source"] = net[source_col].map(lambda x: map_network_node(x, src_type, analysis_id_type, gene_map))
    out["target"] = net[target_col].map(lambda x: map_network_node(x, tgt_type, analysis_id_type, gene_map))

    # merge_grn.py emits `importance`; retain it when its master table is used.
    weight_col = find_col(["weight", "importance", "score", "confidence", "curation_effort"])
    if weight_col is not None:
        out["weight"] = net[weight_col].map(lambda x: abs(safe_float(x, 1.0)))
    else:
        out["weight"] = 1.0

    sign_col = find_col(["sign", "effect", "mor", "mode_of_regulation"])
    stim_col = find_col(["is_stimulation", "stimulation", "consensus_stimulation", "stimulates"])
    inhib_col = find_col(["is_inhibition", "inhibition", "consensus_inhibition", "inhibits"])
    directed_col = find_col(["is_directed", "directed", "consensus_direction"])

    signs = []
    for _, row in net.iterrows():
        sign = 0
        if sign_col is not None:
            raw_value = row[sign_col]
            raw = str(raw_value).strip().lower()
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None:
                sign = 1 if numeric > 0 else (-1 if numeric < 0 else 0)
            elif raw in {"-", "negative", "inhibition", "inhibits", "repression", "down"}:
                sign = -1
            elif raw in {"+", "positive", "activation", "stimulation", "up"}:
                sign = 1
        elif stim_col is not None or inhib_col is not None:
            stim = parse_bool(row[stim_col]) if stim_col is not None else False
            inhib = parse_bool(row[inhib_col]) if inhib_col is not None else False
            if stim and not inhib:
                sign = 1
            elif inhib and not stim:
                sign = -1
        signs.append(sign)

    out["sign"] = signs

    if directed_col is not None:
        directed = net[directed_col].map(parse_bool)
        out = out.loc[directed.values].copy()

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["source", "target", "weight", "sign"])
    out = out[(out["source"] != "") & (out["target"] != "")]
    out = out[out["source"] != out["target"]]
    # An unsigned edge cannot support a signed perturbation prediction.
    out = out[(out["sign"] != 0) & (out["weight"] > 0)]

    out = (
        out.groupby(["source", "target", "sign"], as_index=False)["weight"]
        .sum()
        .sort_values(["source", "target"])
    )
    return out

def read_network(
    network_path: str | Path,
    network_node_id: str = "auto",
    analysis_id_type: str = "symbol",
    gene_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    net = pd.read_csv(network_path, sep=None, engine="python")
    return standardize_network_columns(
        net,
        network_node_id=network_node_id,
        analysis_id_type=analysis_id_type,
        gene_map=gene_map,
    )


def download_omnipath_network(
    resources: Optional[str] = None,
    only_directed: bool = True,
    timeout: int = 120,
    network_node_id: str = "auto",
    analysis_id_type: str = "symbol",
    gene_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """Download signed directed interactions from OmniPath REST API.

    This downloader deliberately avoids the deprecated/unsupported arguments
    genesymbol_resource and consensus_direction-in-fields. It requests gene
    symbols with genesymbols=yes and then standardizes the returned table.

    If you already downloaded a file with source_genesymbol/target_genesymbol,
    prefer passing it with --network and --download_network none.
    """
    base_url = "https://omnipathdb.org/interactions"
    params = {
        "format": "tab",
        "genesymbols": "yes",
    }
    # Keep the API query conservative; unsupported fields cause OmniPath to
    # return "Something is not entirely good" with HTTP 200.
    if resources:
        params["resources"] = resources
    else:
        # Match the user's downloaded file style when possible.
        params["datasets"] = "omnipath"
    params["organisms"] = "9606"

    r = requests.get(base_url, params=params, timeout=timeout)
    if r.status_code != 200 or len(r.text.strip()) == 0:
        params.pop("organisms", None)
        params["organism"] = "9606"
        r = requests.get(base_url, params=params, timeout=timeout)

    if r.status_code != 200:
        raise RuntimeError(f"OmniPath download failed: HTTP {r.status_code}: {r.text[:300]}")

    first_line = r.text.splitlines()[0] if r.text.splitlines() else ""
    if "something is not entirely good" in r.text[:500].lower():
        raise RuntimeError(
            "OmniPath returned an API argument error. Use a locally downloaded network with "
            "source_genesymbol/target_genesymbol and run with --network ... --download_network none. "
            f"First response lines: {r.text[:300]!r}"
        )
    if "source" not in first_line.lower() and "target" not in first_line.lower():
        raise RuntimeError(
            "OmniPath response does not look like an interaction table. "
            f"First line was: {first_line[:200]!r}"
        )

    raw = pd.read_csv(io.StringIO(r.text), sep="\t")
    if raw.empty:
        raise RuntimeError("OmniPath returned an empty table.")

    net = standardize_network_columns(
        raw,
        network_node_id=network_node_id,
        analysis_id_type=analysis_id_type,
        gene_map=gene_map,
    )
    return net


# -----------------------------
# Expression filtering and weighting
# -----------------------------

def filter_and_weight_network(
    net: pd.DataFrame,
    gene_stats: pd.DataFrame,
    min_mean_expr: float,
    min_detected_fraction: float,
    expression_weighting: bool = True,
) -> pd.DataFrame:
    expressed = gene_stats[
        (gene_stats["mean_expr"] >= min_mean_expr)
        & (gene_stats["detected_fraction"] >= min_detected_fraction)
    ].index
    expressed = set(expressed)

    fnet = net[net["source"].isin(expressed) & net["target"].isin(expressed)].copy()
    if fnet.empty:
        raise ValueError(
            "Network is empty after expression filtering. "
            "Try lowering --min_mean_expr or --min_detected_fraction, or use a broader network."
        )

    if expression_weighting:
        mean_expr = gene_stats["mean_expr"].copy()
        q95 = float(np.nanpercentile(mean_expr.values, 95))
        if not np.isfinite(q95) or q95 <= 0:
            q95 = max(float(mean_expr.max()), 1.0)
        activity = (mean_expr / q95).clip(lower=0.0, upper=1.0)
        src_act = fnet["source"].map(activity).fillna(0.0).values
        tgt_act = fnet["target"].map(activity).fillna(0.0).values
        fnet["weight"] = fnet["weight"].astype(float).values * np.sqrt(src_act * tgt_act)
        fnet = fnet[fnet["weight"] > 0].copy()

    # Deduplicate after weighting.
    fnet = fnet.groupby(["source", "target", "sign"], as_index=False)["weight"].sum()
    return fnet


# -----------------------------
# Signed network propagation
# -----------------------------

@dataclass
class PropagationModel:
    genes: List[str]
    gene_to_idx: Dict[str, int]
    transition: sparse.csr_matrix


def build_transition_matrix(net: pd.DataFrame) -> PropagationModel:
    genes = sorted(set(net["source"]).union(set(net["target"])))
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    src = net["source"].map(gene_to_idx).values
    tgt = net["target"].map(gene_to_idx).values
    signed_weight = net["weight"].astype(float).values * net["sign"].astype(float).values

    # Normalize by outgoing absolute weight per source.
    abs_w = np.abs(signed_weight)
    out_sum = np.zeros(len(genes), dtype=float)
    np.add.at(out_sum, src, abs_w)
    out_sum[out_sum == 0] = 1.0
    norm_weight = signed_weight / out_sum[src]

    # transition[target, source] receives source perturbation.
    mat = sparse.coo_matrix((norm_weight, (tgt, src)), shape=(len(genes), len(genes))).tocsr()
    return PropagationModel(genes=genes, gene_to_idx=gene_to_idx, transition=mat)


def propagate_one_ko(
    model: PropagationModel,
    ko_gene: str,
    alpha: float = 0.85,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, bool, int, float]:
    ko_gene = normalize_gene_symbol(ko_gene)
    n = len(model.genes)
    seed = np.zeros(n, dtype=float)
    seed[model.gene_to_idx[ko_gene]] = -1.0

    f = seed.copy()
    last_delta = np.inf
    converged = False

    for it in range(1, max_iter + 1):
        new_f = (1.0 - alpha) * seed + alpha * (model.transition @ f)
        last_delta = float(np.max(np.abs(new_f - f)))
        f = new_f
        if last_delta < tol:
            converged = True
            break

    return f, converged, it, last_delta


def make_ranking(
    ko_gene: str,
    scores: np.ndarray,
    model: PropagationModel,
    gene_stats: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "ko_gene": normalize_gene_symbol(ko_gene),
        "gene": model.genes,
        "signed_impact_score": scores,
    })
    df["abs_impact_score"] = df["signed_impact_score"].abs()
    df["predicted_direction"] = np.where(
        df["signed_impact_score"] > 0, "predicted_increase_after_loss",
        np.where(df["signed_impact_score"] < 0, "predicted_decrease_after_loss", "no_propagated_effect")
    )
    df = df.merge(gene_stats.reset_index(), on="gene", how="left")
    df = df.sort_values("abs_impact_score", ascending=False).reset_index(drop=True)
    df["rank_including_ko"] = np.arange(1, len(df) + 1)

    # Exclude KO gene from useful downstream ranking.
    non_ko = df["gene"] != normalize_gene_symbol(ko_gene)
    ranks = np.full(len(df), np.nan)
    ranks[non_ko.values] = np.arange(1, non_ko.sum() + 1)
    df["rank_excluding_ko"] = ranks
    return df


# -----------------------------
# Enrichr online enrichment
# -----------------------------

DEFAULT_ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
    "WikiPathways_2023_Human",
    "BioPlanet_2019",
    "HumanCyc_2016",
]

METABOLIC_KEYWORDS = [
    "metabolism", "metabolic", "glycolysis", "gluconeogenesis",
    "oxidative phosphorylation", "respiratory chain", "mitochond",
    "fatty acid", "lipid", "cholesterol", "bile acid",
    "amino acid", "glutamine", "glutamate", "tryptophan", "arginine",
    "nucleotide", "purine", "pyrimidine", "one carbon",
    "tca", "citrate cycle", "krebs", "electron transport",
    "xenobiotic", "glutathione", "heme", "peroxisome",
    "steroid", "sphingolipid", "glycerophospholipid",
]


def enrichr_submit_gene_list(genes: Sequence[str], description: str, timeout: int = 60) -> Tuple[int, str]:
    add_url = "https://maayanlab.cloud/Enrichr/addList"
    payload = {
        "list": "\n".join(genes),
        "description": description,
    }
    r = requests.post(add_url, files=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Enrichr addList failed: HTTP {r.status_code}: {r.text[:300]}")
    js = r.json()
    return int(js["userListId"]), str(js.get("shortId", ""))


def enrichr_get_results(user_list_id: int, library: str, timeout: int = 60) -> pd.DataFrame:
    enrich_url = "https://maayanlab.cloud/Enrichr/enrich"
    r = requests.get(
        enrich_url,
        params={"userListId": user_list_id, "backgroundType": library},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Enrichr enrich failed for {library}: HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    rows = data.get(library, [])
    if not rows:
        return pd.DataFrame(columns=[
            "library", "rank", "term", "p_value", "z_score",
            "combined_score", "overlap_genes", "adjusted_p_value"
        ])

    out_rows = []
    for row in rows:
        # Enrichr row format commonly:
        # [rank, term, pvalue, zscore, combined_score, overlapping_genes, adjusted_pvalue, old_p, old_adj]
        out_rows.append({
            "library": library,
            "rank": row[0] if len(row) > 0 else np.nan,
            "term": row[1] if len(row) > 1 else "",
            "p_value": row[2] if len(row) > 2 else np.nan,
            "z_score": row[3] if len(row) > 3 else np.nan,
            "combined_score": row[4] if len(row) > 4 else np.nan,
            "overlap_genes": ";".join(row[5]) if len(row) > 5 and isinstance(row[5], list) else (row[5] if len(row) > 5 else ""),
            "adjusted_p_value": row[6] if len(row) > 6 else np.nan,
        })
    return pd.DataFrame(out_rows)


def run_enrichr(
    genes: Sequence[str],
    ko_gene: str,
    libraries: Sequence[str],
    timeout: int = 60,
    sleep_sec: float = 0.2,
) -> pd.DataFrame:
    genes = [normalize_gene_symbol(g) for g in genes if str(g).strip()]
    if len(genes) == 0:
        return pd.DataFrame()

    uid, sid = enrichr_submit_gene_list(genes, f"{ko_gene}_top_affected_genes", timeout=timeout)
    all_res = []
    for lib in libraries:
        try:
            res = enrichr_get_results(uid, lib, timeout=timeout)
            if not res.empty:
                res.insert(0, "ko_gene", normalize_gene_symbol(ko_gene))
                res.insert(1, "n_input_genes", len(genes))
                all_res.append(res)
        except Exception as e:
            all_res.append(pd.DataFrame([{
                "ko_gene": normalize_gene_symbol(ko_gene),
                "n_input_genes": len(genes),
                "library": lib,
                "rank": np.nan,
                "term": f"ERROR: {e}",
                "p_value": np.nan,
                "z_score": np.nan,
                "combined_score": np.nan,
                "overlap_genes": "",
                "adjusted_p_value": np.nan,
            }]))
        time.sleep(sleep_sec)
    if all_res:
        return pd.concat(all_res, ignore_index=True)
    return pd.DataFrame()


# -----------------------------
# Offline GMT enrichment
# -----------------------------

def read_gmt(gmt_path: str | Path) -> Dict[str, set]:
    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = {normalize_gene_symbol(g) for g in parts[2:] if g.strip()}
            if genes:
                gene_sets[term] = genes
    return gene_sets


def run_offline_gmt_enrichment(
    genes: Sequence[str],
    background: Sequence[str],
    gene_sets: Dict[str, set],
    ko_gene: str,
    min_overlap: int = 2,
) -> pd.DataFrame:
    query = {normalize_gene_symbol(g) for g in genes}
    bg = {normalize_gene_symbol(g) for g in background}
    query = query & bg
    M = len(bg)
    N = len(query)
    rows = []
    if M == 0 or N == 0:
        return pd.DataFrame()

    for term, gs in gene_sets.items():
        gs = gs & bg
        K = len(gs)
        if K == 0:
            continue
        overlap = query & gs
        x = len(overlap)
        # P(X >= x)
        p = float(hypergeom.sf(x - 1, M, K, N))
        rows.append({
            "ko_gene": normalize_gene_symbol(ko_gene),
            "library": "offline_gmt",
            "term": term,
            "p_value": p,
            "overlap_count": x,
            "set_size_in_background": K,
            "query_size": N,
            "overlap_genes": ";".join(sorted(overlap)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["adjusted_p_value"] = bh_fdr(out["p_value"].values)
    out = out[out["overlap_count"] >= min_overlap].copy()
    out = out.sort_values(["adjusted_p_value", "p_value", "term"]).reset_index(drop=True)
    return out


def filter_metabolic_terms(enrich_df: pd.DataFrame) -> pd.DataFrame:
    if enrich_df is None or enrich_df.empty or "term" not in enrich_df.columns:
        return pd.DataFrame()
    pattern = "|".join([kw.replace(" ", r"\s+") for kw in METABOLIC_KEYWORDS])
    mask = enrich_df["term"].astype(str).str.lower().str.contains(pattern, regex=True, na=False)
    return enrich_df.loc[mask].copy()


# -----------------------------
# Main workflow
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Small-sample bulk RNA prior-network single-gene virtual KO impact ranking."
    )
    ap.add_argument("--expr", required=True, help="Expression matrix: genes in rows, samples in columns.")
    ap.add_argument("--ko", required=True, help="Text file with one target gene symbol per line.")
    ap.add_argument("--network", required=True, help="Signed prior network, e.g. merged_grn_master.tsv from merge_grn.py.")
    ap.add_argument("--outdir", default="bulk_ko_results", help="Output folder (default: bulk_ko_results).")

    ap.add_argument("--gene_col", default="GeneID", help="Gene ID column (default: GeneID; if absent, uses the first column).")
    ap.add_argument("--analysis_id_type", default="symbol", choices=["symbol", "ensembl"],
                    help="Identifier namespace used internally for expression, KO list, and network. Default: symbol.")
    ap.add_argument("--expr_id_type", default="auto", choices=["auto", "symbol", "ensembl"],
                    help="Identifier type of the expression first column. Default: auto.")
    ap.add_argument("--ko_id_type", default="auto", choices=["auto", "symbol", "ensembl"],
                    help="Identifier type of KO genes. Default: auto.")
    ap.add_argument("--network_node_id", default="auto", choices=["auto", "symbol", "ensembl", "uniprot"],
                    help="Identifier type in network source/target if no gene symbol columns exist. Default: auto.")
    ap.add_argument("--gene_map", default=None,
                    help="Optional gene ID mapping table. Recommended columns: symbol, ensembl, uniprot. Required for Ensembl expression -> symbol or UniProt network -> symbol.")
    ap.add_argument("--group_file", default=None,
                    help="Optional headerless 2-column file: sample ID, group (e.g. PE/Control).")
    ap.add_argument("--group", default=None,
                    help="Optional group to analyze alone; omit to use all expression samples.")

    ap.add_argument("--min_mean_expr", type=float, default=1.0,
                    help="Minimum mean expression to keep a gene. For log2(TPM+1), 1 roughly means TPM>=1.")
    ap.add_argument("--min_detected_fraction", type=float, default=0.5,
                    help="Minimum fraction of samples with expression > detected_threshold.")
    ap.add_argument("--detected_threshold", type=float, default=0.0,
                    help="Threshold for detected_fraction calculation. Default: 0.")
    ap.add_argument("--disable_expression_weighting", action="store_true",
                    help="Disable expression-based edge weighting after filtering.")

    ap.add_argument("--alpha", type=float, default=0.85,
                    help="Propagation strength, 0-1. Larger values propagate farther. Default: 0.85.")
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--top_n", type=int, default=200,
                    help="Number of top affected genes to output per KO.")
    ap.add_argument("--enrich_top_n", type=int, default=100,
                    help="Number of top affected genes used for enrichment.")
    ap.add_argument("--min_abs_impact", type=float, default=1e-12,
                    help="Minimum absolute propagated impact score to be considered affected; zero/noise scores are excluded.")
    ap.add_argument("--skip_enrichr", action="store_true",
                    help="Skip online Enrichr analysis.")
    ap.add_argument("--enrichr_libraries", default=",".join(DEFAULT_ENRICHR_LIBRARIES),
                    help="Comma-separated Enrichr libraries.")
    ap.add_argument("--offline_gmt", default=None,
                    help="Optional GMT file for offline hypergeometric enrichment.")
    ap.add_argument("--min_overlap", type=int, default=2,
                    help="Minimum overlap for offline GMT enrichment.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not (0 < args.alpha < 1):
        raise ValueError("--alpha must be between 0 and 1.")
    if bool(args.group_file) != bool(args.group):
        raise ValueError("To select one group, provide both --group_file and --group. Omit both to use all samples.")

    outdir = mkdir(args.outdir)
    ranking_dir = mkdir(outdir / "rankings")
    top_dir = mkdir(outdir / "top_genes")
    enrich_dir = mkdir(outdir / "enrichment")

    log_lines = []
    log_lines.append("bulk_ko_auto.py QC report")
    log_lines.append("=========================")

    # Optional ID mapping
    gene_map = read_gene_map(args.gene_map)
    if gene_map is not None:
        log_lines.append(f"Gene ID map loaded: {args.gene_map}")

    # Expression
    expr = read_expression(
        args.expr,
        gene_col=args.gene_col,
        expr_id_type=args.expr_id_type,
        analysis_id_type=args.analysis_id_type,
        gene_map=gene_map,
        unmapped_out=outdir / "unmapped_expression_ids.tsv",
    )
    if args.group_file:
        meta_df = pd.read_csv(args.group_file, sep=r"\s+", engine="python", header=None,
                              usecols=[0, 1], names=["sample", "group"], dtype=str)
        meta_df["sample"] = meta_df["sample"].str.strip()
        meta_df["group"] = meta_df["group"].str.strip()
        selected = set(meta_df.loc[meta_df["group"] == str(args.group), "sample"])
        selected_cols = [c for c in expr.columns if str(c) in selected]
        if not selected_cols:
            raise ValueError(f"No expression columns matched group {args.group!r}; check sample IDs and group file.")
        expr = expr[selected_cols]
    gene_stats = compute_gene_stats(expr, detected_threshold=args.detected_threshold)
    gene_stats.to_csv(outdir / "gene_expression_stats.tsv", sep="\t")
    log_lines.append(f"Analysis ID type: {args.analysis_id_type}")
    log_lines.append(f"Expression genes after ID normalization: {expr.shape[0]}")
    log_lines.append(f"Expression samples used: {expr.shape[1]}")
    if args.group is not None:
        log_lines.append(f"Expression group selected: {args.group}")

    # KO list
    ko_genes, ko_unmapped = read_ko_genes(
        args.ko,
        ko_id_type=args.ko_id_type,
        analysis_id_type=args.analysis_id_type,
        gene_map=gene_map,
    )
    if ko_unmapped:
        pd.DataFrame({"unmapped_ko_gene": ko_unmapped}).to_csv(outdir / "unmapped_ko_ids.tsv", sep="\t", index=False)
    if len(ko_genes) == 0:
        raise ValueError("KO list is empty after ID normalization/mapping.")
    log_lines.append(f"Requested KO genes after ID normalization: {len(ko_genes)}")
    if ko_unmapped:
        log_lines.append(f"Unmapped KO IDs: {len(ko_unmapped)}")

    # Network
    net = read_network(
        args.network,
        network_node_id=args.network_node_id,
        analysis_id_type=args.analysis_id_type,
        gene_map=gene_map,
    )
    log_lines.append(f"Network loaded from local file: {args.network}")

    net.to_csv(outdir / "prior_network_raw_standardized.tsv", sep="\t", index=False)
    raw_nodes = set(net.source).union(set(net.target))
    expr_nodes = set(expr.index)
    overlap_nodes = sorted(expr_nodes & raw_nodes)
    pd.DataFrame({"overlap_gene": overlap_nodes}).to_csv(outdir / "expression_network_overlap_genes.tsv", sep="\t", index=False)
    log_lines.append(f"Raw standardized network edges: {len(net)}")
    log_lines.append(f"Raw standardized network genes: {len(raw_nodes)}")
    log_lines.append(f"Expression-network overlapping genes: {len(overlap_nodes)}")
    if len(overlap_nodes) < 100:
        log_lines.append("WARNING: Very small expression-network overlap. Check ID types and --gene_map.")

    # Filter and weight
    fnet = filter_and_weight_network(
        net,
        gene_stats=gene_stats,
        min_mean_expr=args.min_mean_expr,
        min_detected_fraction=args.min_detected_fraction,
        expression_weighting=not args.disable_expression_weighting,
    )
    fnet.to_csv(outdir / "prior_network_expression_filtered.tsv", sep="\t", index=False)
    log_lines.append(f"Filtered network edges: {len(fnet)}")
    log_lines.append(f"Filtered network genes: {len(set(fnet.source).union(set(fnet.target)))}")
    log_lines.append(f"Expression weighting: {not args.disable_expression_weighting}")

    # Build propagation model
    model = build_transition_matrix(fnet)
    log_lines.append(f"Propagation model genes: {len(model.genes)}")
    log_lines.append(f"Propagation model nonzero edges: {model.transition.nnz}")

    missing = [g for g in ko_genes if g not in model.gene_to_idx]
    valid_kos = [g for g in ko_genes if g in model.gene_to_idx]
    log_lines.append(f"Valid KO genes in model: {len(valid_kos)}")
    if missing:
        log_lines.append(f"Missing KO genes after network/expression filtering: {','.join(missing)}")

    # Offline GMT
    gmt_sets = None
    if args.offline_gmt:
        gmt_sets = read_gmt(args.offline_gmt)
        log_lines.append(f"Offline GMT gene sets loaded: {len(gmt_sets)}")

    enrichr_libs = [x.strip() for x in args.enrichr_libraries.split(",") if x.strip()]

    all_top = []
    all_enrich = []
    all_metabolic = []

    for ko in valid_kos:
        scores, converged, n_iter, last_delta = propagate_one_ko(
            model, ko_gene=ko, alpha=args.alpha, max_iter=args.max_iter, tol=args.tol
        )
        rank_df = make_ranking(ko, scores, model, gene_stats)
        rank_df.to_csv(ranking_dir / f"{ko}.ranking.tsv", sep="\t", index=False)

        top_df = (
            rank_df[(rank_df["gene"] != ko) & (rank_df["abs_impact_score"] > args.min_abs_impact)]
            .sort_values("abs_impact_score", ascending=False)
            .head(args.top_n)
            .copy()
        )
        top_df.to_csv(top_dir / f"{ko}.top{args.top_n}.tsv", sep="\t", index=False)
        all_top.append(top_df)

        log_lines.append(
            f"KO {ko}: converged={converged}, iterations={n_iter}, last_delta={last_delta:.3e}"
        )

        enrich_genes = top_df["gene"].head(args.enrich_top_n).tolist()

        # Online Enrichr
        if not enrich_genes:
            log_lines.append(f"KO {ko}: no genes exceeded min_abs_impact={args.min_abs_impact}; enrichment skipped.")
        elif not args.skip_enrichr:
            try:
                er = run_enrichr(enrich_genes, ko_gene=ko, libraries=enrichr_libs)
                if not er.empty:
                    er.to_csv(enrich_dir / f"{ko}.enrichr.tsv", sep="\t", index=False)
                    all_enrich.append(er)
                    met = filter_metabolic_terms(er)
                    if not met.empty:
                        met.to_csv(enrich_dir / f"{ko}.metabolic_enrichr.tsv", sep="\t", index=False)
                        all_metabolic.append(met)
            except Exception as e:
                err = pd.DataFrame([{
                    "ko_gene": ko,
                    "error": f"Enrichr failed: {e}",
                }])
                err.to_csv(enrich_dir / f"{ko}.enrichr.ERROR.tsv", sep="\t", index=False)
                log_lines.append(f"KO {ko}: Enrichr failed: {e}")

        # Offline GMT
        if gmt_sets is not None and enrich_genes:
            offline = run_offline_gmt_enrichment(
                enrich_genes,
                background=model.genes,
                gene_sets=gmt_sets,
                ko_gene=ko,
                min_overlap=args.min_overlap,
            )
            if not offline.empty:
                offline.to_csv(enrich_dir / f"{ko}.offline_gmt.tsv", sep="\t", index=False)
                all_enrich.append(offline)
                met = filter_metabolic_terms(offline)
                if not met.empty:
                    met.to_csv(enrich_dir / f"{ko}.metabolic_offline_gmt.tsv", sep="\t", index=False)
                    all_metabolic.append(met)

    if all_top:
        pd.concat(all_top, ignore_index=True).to_csv(
            outdir / f"ALL_KO.top{args.top_n}.tsv", sep="\t", index=False
        )

    if all_enrich:
        pd.concat(all_enrich, ignore_index=True).to_csv(
            outdir / "ALL_KO.enrichment.tsv", sep="\t", index=False
        )

    if all_metabolic:
        pd.concat(all_metabolic, ignore_index=True).to_csv(
            outdir / "ALL_KO.metabolic_enrichment.tsv", sep="\t", index=False
        )
    else:
        # Create an empty but informative file.
        pd.DataFrame(columns=[
            "ko_gene", "library", "term", "p_value", "adjusted_p_value", "overlap_genes"
        ]).to_csv(outdir / "ALL_KO.metabolic_enrichment.tsv", sep="\t", index=False)

    # Run metadata
    meta = {
        "expr": str(args.expr),
        "analysis_id_type": args.analysis_id_type,
        "expr_id_type": args.expr_id_type,
        "ko_id_type": args.ko_id_type,
        "network_node_id": args.network_node_id,
        "gene_map": str(args.gene_map) if args.gene_map else None,
        "group_file": str(args.group_file) if args.group_file else None,
        "group": args.group,
        "network": str(args.network) if args.network else None,
        "min_mean_expr": args.min_mean_expr,
        "min_detected_fraction": args.min_detected_fraction,
        "detected_threshold": args.detected_threshold,
        "expression_weighting": not args.disable_expression_weighting,
        "alpha": args.alpha,
        "top_n": args.top_n,
        "enrich_top_n": args.enrich_top_n,
        "min_abs_impact": args.min_abs_impact,
        "enrichr_libraries": enrichr_libs,
        "offline_gmt": str(args.offline_gmt) if args.offline_gmt else None,
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(outdir / "qc_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"Done. Results written to: {outdir}")


if __name__ == "__main__":
    main()
