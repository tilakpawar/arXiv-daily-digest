#!/usr/bin/env python3
"""
arXiv Digest — fully local daily reader for eclipsing binaries & asteroseismology.
Uses a living taxonomy (taxonomy.json) and a paper database (papers.json) that
Qwen learns from over time, improving sub-topic classification on every run.

Usage:
  python3 arxiv_digest.py                      normal run
  python3 arxiv_digest.py -v                   verbose (DEBUG) logging
  python3 arxiv_digest.py --dry-run            fetch + filter only, no LLM
  python3 arxiv_digest.py --no-triage          skip LLM relevance check (keyword only)
  python3 arxiv_digest.py --days 7             widen the lookback window
  python3 arxiv_digest.py --reset-seen         forget all previously seen papers
  python3 arxiv_digest.py --limit 5            cap papers summarised this run
  python3 arxiv_digest.py --save-feed          dump raw arXiv feed to feed_debug.xml
  python3 arxiv_digest.py --topics "Pulsating EBs"   restrict to sub-topic(s)
  python3 arxiv_digest.py --list-topics        show taxonomy and exit
  python3 arxiv_digest.py --expand-taxonomy    ask Qwen to grow the taxonomy from
                                               recent arXiv papers (run occasionally)

Each stage logs how many papers entered and left it, so a "0 papers" result tells
you exactly where the funnel emptied.

Files written next to this script:
  taxonomy.json   living topic/sub-topic tree — hand-seeded, Qwen-expanded
  papers.json     DB of every classified paper (few-shot context for triage)
  seen.json                  set of arXiv IDs already processed
  taxonomy_embeddings.json   cached sub-topic embedding vectors (auto-rebuilt when taxonomy changes)
  reports/        one HTML file per run + index.html
"""

import argparse
import copy
import html
import json
import logging
import re
import time
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import threading
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OLLAMA_MODEL  = "qwen2.5:7b"
OLLAMA_URL    = "http://localhost:11434"

CATEGORIES    = ["astro-ph.SR"]
MAX_FETCH     = 300
LOOKBACK_DAYS = 3

# How many recent classified papers to show Qwen as few-shot triage examples
TRIAGE_EXAMPLES   = 6
# How many papers to triage in one Ollama call.
# One call per paper: N papers × ~30s = slow.
# One call for N papers: one prompt, one JSON array back = ~35-40s total.
# Sweet spot is 8-12; beyond that the model loses accuracy on later items.
TRIAGE_BATCH_SIZE = 10
# chars fed to the LLM
INTRO_CHARS       = 3500
CONCLUSION_CHARS  = 4000
METHODS_CHARS     = 3000
ABSTRACT_CHARS    = 2500
NUM_CTX           = 8192

# OAI-PMH endpoint — the official bulk-harvesting interface.
# Unlike the REST API (export.arxiv.org/api/query) this endpoint has no
# per-request rate limits and is designed for exactly this kind of daily
# incremental harvest.  See https://info.arxiv.org/help/oa/index.html
OAI_PMH_URL = "https://oaipmh.arxiv.org/oai"
USER_AGENT  = "arxiv-digest/1.0 (mailto:tpawar@villanova.edu)"
HTTP_TIMEOUT = (10, 90)

# Polite delay between OAI-PMH requests (resumption-token pages).
# The endpoint has no hard rate limit but a small sleep is courteous.
OAI_PAGE_DELAY = 1.0
# Shared state so concurrent fetch_fulltext threads queue politely
_ARXIV_FETCH_LOCK = threading.Lock()
_ARXIV_LAST_REQ   = [0.0]  # mutable singleton

HERE          = Path(__file__).resolve().parent
REPORT_DIR    = HERE / "reports"
SEEN_FILE     = HERE / "seen.json"
TAXONOMY_FILE = HERE / "taxonomy.json"
PAPERS_FILE   = HERE / "papers.json"
EMBED_FILE    = HERE / "taxonomy_embeddings.json"

# Background text representing generic astro-ph.SR content — the "null model".
# The lift score is cosine(paper, subtopic) - cosine(paper, background).
# Papers genuinely about EBs/asteroseismology will have positive lift;
# papers that mention related words incidentally will have lift near zero.
EMBED_BACKGROUND = (
    "General stellar astrophysics paper in astro-ph.SR covering topics such as "
    "stellar evolution, stellar atmospheres, stellar winds, supernovae, star "
    "formation, circumstellar matter, solar physics, magnetic activity, "
    "spectroscopy, photometry, radial velocity, interstellar medium, "
    "exoplanets, brown dwarfs, white dwarfs, neutron stars, gamma-ray bursts, "
    "nucleosynthesis, stellar populations, galactic structure."
)

# Embedding model — must be pulled in Ollama: `ollama pull nomic-embed-text`
# It is tiny (~274 MB) and ~100x faster than qwen2.5:7b for embeddings.
# Falls back to qwen2.5:7b if not available.
EMBED_MODEL   = "nomic-embed-text"

# Thresholds applied to the *lift* score (subtopic similarity MINUS background
# similarity).  This is much more discriminating than raw cosine similarity
# because all astro-ph.SR papers look similar to each other in embedding space.
# Lift > 0 means the paper is closer to the subtopic than to generic astro content.
#
#   lift >= EMBED_ACCEPT  → accepted without Ollama (clear hit)
#   EMBED_REVIEW <= lift < EMBED_ACCEPT → sent to Ollama triage (borderline)
#   lift <  EMBED_REVIEW  → dropped (clear miss)
#
# Run --calibrate to see the lift distribution and get suggested values.
EMBED_ACCEPT  = -999   # lift above background for confident accept
EMBED_REVIEW  = -999   # lift above background for borderline review

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:
    from requests.packages.urllib3.util.retry import Retry
_retry = Retry(total=4, connect=4, read=3, backoff_factor=1.5,
               status_forcelist=(500, 502, 503, 504),
               allowed_methods=frozenset(["GET", "POST"]))
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
SESSION.mount("http://",  HTTPAdapter(max_retries=_retry))

log = logging.getLogger("arxiv-digest")


# ---------------------------------------------------------------------------
# SEED TAXONOMY — written to taxonomy.json on first run if file absent.
# Structure:  { topic: { desc, subtopics: { name: {desc, keywords} } } }
# ---------------------------------------------------------------------------
SEED_TAXONOMY = {
    "Eclipsing Binaries": {
        "desc": "Stars whose orbital plane is aligned so components transit each other, "
                "enabling precise masses, radii, and light-curve modelling.",
        "subtopics": {
            "Detached & semi-detached EBs": {
                "desc": "Well-separated components, Roche-lobe geometry, precise absolute parameters.",
                "keywords": [
                    "detached binar", "semi-detached", "eclipsing binar", "eclipsing system",
                    "absolute parameter", "mass ratio", "roche lobe", "light curve solution",
                    "radial velocit", "double-lined", "spectroscopic binar",
                ]
            },
            "Contact & overcontact EBs": {
                "desc": "W UMa-type systems sharing a common envelope.",
                "keywords": [
                    "contact binar", "overcontact", "w uma", "w ursae majoris",
                    "common envelope", "fill factor", "shallow contact",
                ]
            },
            "Pulsating stars in EBs": {
                "desc": "EBs hosting intrinsic pulsators: delta Sct, gamma Dor, "
                        "RR Lyr, Cepheid, sdBV, roAp inside eclipsing or SB2 systems.",
                "keywords": [
                    "pulsating", "pulsation", "oscillat", "delta scuti", "delta sct",
                    "gamma dor", "gamma doradus", "rr lyr", "cepheid", "sdbv",
                    "roap", "tidally excited", "tidal pulsation", "heartbeat star",
                    "resonance locking", "p-mode", "g-mode", "mixed mode",
                    "eclipsing binar", "binary pulsator",
                ]
            },
            "Heartbeat & tidally distorted EBs": {
                "desc": "Eccentric binaries with tidal distortion near periastron; "
                        "tidally excited oscillations.",
                "keywords": [
                    "heartbeat", "tidally distort", "tidal distortion", "eccentric binar",
                    "periastron", "tidal excitation", "tidal oscillat", "ellipsoidal",
                ]
            },
            "EB light-curve & radial-velocity modelling": {
                "desc": "Codes and methods: PHOEBE, ELC, JKTEBOP, WD, etc.",
                "keywords": [
                    "phoebe", "jktebop", "elc code", "wd code", "wilson-devinney",
                    "light curve model", "lc modelling", "rv curve", "synthetic light curve",
                    "light curve fitting", "binary modelling", "eclipsing model",
                ]
            },
            "ML & statistical classification of EBs": {
                "desc": "Machine learning, neural networks, random forests used to "
                        "identify or classify EBs in survey data.",
                "keywords": [
                    "machine learning", "neural network", "random forest", "deep learning",
                    "convolutional network", "cnn", "gradient boost", "xgboost",
                    "eclipsing binary classif", "binary classif", "photometric classif",
                    "supervised learning", "unsupervised", "clustering", "variability classif",
                ]
            },
            "EB catalogue & survey studies": {
                "desc": "Large-scale EB catalogues from Kepler, TESS, OGLE, Gaia, etc.",
                "keywords": [
                    "kepler eb", "tess eb", "ogle eb", "gaia eb", "catalogue",
                    "catalog", "survey", "photometric survey", "all-sky", "eclipsing binar",
                    "period catalog", "eb list", "tess", "transiting exoplanet survey satellite",
                ]
            },
            "Third bodies & multiple systems": {
                "desc": "Tertiary companions found via eclipse timing variations (ETVs) or "
                        "spectroscopy; hierarchical triples.",
                "keywords": [
                    "third body", "tertiary", "eclipse timing", "etv", "o-c diagram",
                    "timing variation", "hierarchical triple", "quadruple", "outer orbit",
                    "light travel time",
                ]
            },
            "Mass transfer & evolution in EBs": {
                "desc": "Algol-type systems, mass transfer, circularisation, apsidal motion.",
                "keywords": [
                    "mass transfer", "algol", "apsidal motion", "apsidal precession",
                    "circularisation", "circularization", "spin-orbit", "tidal synchron",
                    "evolutionary track", "isochrone", "binary evolution",
                ]
            },
        }
    },
    "Asteroseismology": {
        "desc": "Probing stellar interiors through oscillation frequencies.",
        "subtopics": {
            "Solar-like oscillations & red giants": {
                "desc": "Stochastically excited p-modes and mixed modes in solar-type "
                        "stars and red giants; nu_max, Delta_nu scaling.",
                "keywords": [
                    "solar-like oscillat", "red giant", "subgiant", "delta nu",
                    "large separation", "nu max", "mixed mode", "dipole mode",
                    "asymptotic relation", "stochastic excitation", "kepler asteroseismol",
                    "tess asteroseismol",
                ]
            },
            "Classical pulsators": {
                "desc": "Instability-strip pulsators: delta Sct, gamma Dor, RR Lyr, "
                        "Cepheids, roAp, SdB/SdO pulsators.",
                "keywords": [
                    "delta scuti", "delta sct", "gamma dor", "gamma doradus",
                    "rr lyrae", "cepheid", "roap", "rapidly oscillating ap",
                    "sdb pulsator", "v361 hya", "v1093 her", "instability strip",
                ]
            },
            "Frequency analysis & mode identification": {
                "desc": "Period spacing, period-luminosity, echelle diagrams, "
                        "prewhitening, mode identification.",
                "keywords": [
                    "frequency analysis", "period spacing", "echelle diagram",
                    "prewhitening", "iterative prewhitening", "mode identification",
                    "period-luminosity", "p-l relation", "fourier analysis",
                    "combination frequency", "harmonic",
                ]
            },
            "Rotation & angular momentum in asteroseismology": {
                "desc": "Internal rotation profiles, angular momentum transport, "
                        "magnetic fields inferred from splittings.",
                "keywords": [
                    "rotation profile", "internal rotation", "angular momentum",
                    "rotational splitting", "core rotation", "envelope rotation",
                    "magnetic field asteroseismol", "differential rotation",
                ]
            },
            "Stellar structure & modelling": {
                "desc": "Grid modelling, stellar evolution codes, convective overshoot, "
                        "opacity, equation of state.",
                "keywords": [
                    "stellar model", "stellar structure", "evolutionary model",
                    "convective overshoot", "convective penetration", "opacity",
                    "equation of state", "mesa", "cestam", "garstec", "astec",
                    "seismic constraint", "model grid",
                ]
            },
            "Space photometry & instrumentation": {
                "desc": "Kepler, K2, TESS, PLATO pipelines and data products for "
                        "asteroseismology.",
                "keywords": [
                    "kepler photometr", "k2 photometr", "tess photometr", "plato",
                    "long cadence", "short cadence", "two-minute cadence",
                    "twenty-second", "pixel file", "lightkurve", "background correction", "tess", "tess sector", "tess light curve",
                ]
            },
        }
    }
}


# ---------------------------------------------------------------------------
# TAXONOMY I/O
# ---------------------------------------------------------------------------
def load_taxonomy():
    if TAXONOMY_FILE.exists():
        tax = json.loads(TAXONOMY_FILE.read_text())
        log.debug("Loaded taxonomy from %s (%d topics)", TAXONOMY_FILE, len(tax))
        return tax
    log.info("taxonomy.json not found — writing seed taxonomy")
    save_taxonomy(SEED_TAXONOMY)
    return copy.deepcopy(SEED_TAXONOMY)


def save_taxonomy(tax):
    TAXONOMY_FILE.write_text(json.dumps(tax, indent=2, ensure_ascii=False))


def all_subtopics(tax):
    """Flat dict: subtopic_name -> {desc, keywords, parent_topic}"""
    out = {}
    for tname, t in tax.items():
        for sname, s in t.get("subtopics", {}).items():
            out[sname] = {**s, "parent": tname}
    return out


def all_keywords(tax):
    """Flat set of all keywords across the whole taxonomy."""
    kws = set()
    for t in tax.values():
        for s in t.get("subtopics", {}).values():
            kws.update(s.get("keywords", []))
    return kws


# ---------------------------------------------------------------------------
# SEED TAXONOMY FROM AUTHOR  (--seed-author "Surname Forename")
# ---------------------------------------------------------------------------
def seed_from_author(author_name, tax, lookback_days=365):
    """
    Harvest recent astro-ph.SR papers via OAI-PMH (same bulk endpoint as
    fetch_feed), filter client-side by author name, then ask Qwen to suggest
    keyword/subtopic additions to the taxonomy based on those papers.

    No extra API calls — reuses the existing SESSION + OAI-PMH pipeline.
    """
    log.info("SEED FROM AUTHOR — harvesting last %dd of astro-ph.SR …", lookback_days)

    # Temporarily raise MAX_FETCH for this wide harvest
    import xml.etree.ElementTree as ET
    cutoff   = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    from_str = cutoff.strftime("%Y-%m-%d")
    oai_set  = _oai_set_for_category("astro-ph.SR")

    params   = {
        "verb":           "ListRecords",
        "metadataPrefix": "arXiv",
        "set":            oai_set,
        "from":           from_str,
    }

    # Normalise author name for fuzzy matching:
    # "Pawar T" / "T Pawar" / "Pawar, T" all reduce to a comparable set of tokens
    name_tokens = set(re.split(r"[\s,\.]+", author_name.lower())) - {""}

    all_raw  = []
    page     = 0
    MAX_AUTHOR_FETCH = 2000   # pages through up to 2000 records bulk

    log.info("  OAI-PMH bulk harvest | set=%s | from=%s | cap=%d",
             oai_set, from_str, MAX_AUTHOR_FETCH)

    while len(all_raw) < MAX_AUTHOR_FETCH:
        url = OAI_PMH_URL + "?" + requests.compat.urlencode(params)
        log.debug("  OAI page %d — GET %s", page, url)
        try:
            resp = SESSION.get(url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("OAI-PMH request failed: %s", e)
            break

        records, token = _oai_parse_records(resp.text)
        log.debug("  page %d: %d records", page, len(records))
        all_raw.extend(records)

        if not token:
            break
        params = {"verb": "ListRecords", "resumptionToken": token}
        page  += 1
        time.sleep(OAI_PAGE_DELAY)

    log.info("  %d total records harvested — filtering by author %r …",
             len(all_raw), author_name)

    # Client-side author filter: any name token overlap is a candidate hit.
    # e.g. name_tokens={"pawar","t"} matches author "T Pawar" or "Pawar Tushar"
    matched = []
    for r in all_raw:
        for a in r.get("authors", []):
            a_tokens = set(re.split(r"[\s,\.]+", a.lower())) - {""}
            if name_tokens & a_tokens:   # non-empty intersection
                matched.append(r)
                break

    log.info("  %d papers matched author %r", len(matched), author_name)
    if not matched:
        log.warning("No papers found for author %r — check spelling or widen --days.",
                    author_name)
        return tax

    # Convert to the same paper dict shape the rest of the pipeline uses
    papers = []
    for r in matched:
        base_id = r["base_id"]
        papers.append({
            "base_id":   base_id,
            "title":     r["title"],
            "abstract":  r["abstract"],
            "authors":   r["authors"],
            "published": r["date_str"],
            "abs_url":   f"https://arxiv.org/abs/{base_id}",
        })

    # Show what we found before passing to Qwen
    log.info("  Papers found:")
    for p in papers:
        log.info("    [%s] %s", p["published"], p["title"][:70])

    if not check_ollama():
        log.error("Ollama unreachable — cannot seed taxonomy without LLM.")
        return tax

    # Build taxonomy summary and paper block for Qwen — same pattern as
    # expand_taxonomy()
    tax_summary = []
    for tname, t in tax.items():
        tax_summary.append(f"Topic: {tname}")
        for sname, s in t.get("subtopics", {}).items():
            kws = ", ".join(s.get("keywords", [])[:10])
            tax_summary.append(f"  Sub-topic: {sname}  [keywords: {kws}]")
    tax_str = "\n".join(tax_summary)

    paper_block = "\n\n".join(
        f"Title: {p['title']}\nAbstract: {p['abstract'][:600]}"
        for p in papers[:40]   # cap at 40 to stay within context
    )

    prompt = (
        "You are an expert in stellar astrophysics, eclipsing binaries, and asteroseismology.\n\n"
        "A researcher's own papers are listed below. Your job is to extract keywords and "
        "phrases from their titles and abstracts that would help a daily arXiv digest "
        "find papers similar to their work.\n\n"
        "Current taxonomy:\n"
        f"{tax_str}\n\n"
        "Researcher's papers:\n\n"
        f"{paper_block}\n\n"
        "Respond with JSON only — suggest keyword additions to existing sub-topics, "
        "and new sub-topics if the researcher works on something not yet covered:\n"
        "{\n"
        '  "keyword_additions": [\n'
        '    {"subtopic": "<existing sub-topic name>", "keywords": ["kw1", "kw2", ...]},\n'
        '    ...\n'
        '  ],\n'
        '  "new_subtopics": [\n'
        '    {"parent_topic": "<existing topic name>", "name": "<sub-topic name>",\n'
        '     "desc": "<one sentence>", "keywords": ["kw1", ...]},\n'
        '    ...\n'
        '  ]\n'
        '}'
    )

    try:
        raw      = ollama(prompt, json_mode=True, num_predict=1500)
        proposals = json.loads(raw)
    except Exception as e:
        log.error("Qwen author-seed failed: %s", e)
        return tax

    # Merge proposals — identical logic to expand_taxonomy()
    new_tax        = copy.deepcopy(tax)
    n_new_subtopics = 0
    n_new_keywords  = 0

    for ka in proposals.get("keyword_additions", []):
        subtopic = ka.get("subtopic", "").strip()
        new_kws  = [k.lower() for k in ka.get("keywords", [])]
        for tname, t in new_tax.items():
            if subtopic in t.get("subtopics", {}):
                existing = set(t["subtopics"][subtopic].get("keywords", []))
                added    = [k for k in new_kws if k not in existing]
                t["subtopics"][subtopic].setdefault("keywords", []).extend(added)
                if added:
                    log.info("  + keywords for [%s]: %s", subtopic, added)
                    n_new_keywords += len(added)
                break

    for ns in proposals.get("new_subtopics", []):
        parent = ns.get("parent_topic", "").strip()
        name   = ns.get("name", "").strip()
        if not name or not parent:
            continue
        matched_parent = next(
            (k for k in new_tax if k.lower() == parent.lower()), None)
        if matched_parent is None:
            log.debug("  author-seed: unknown parent %r — skipping", parent)
            continue
        if name not in new_tax[matched_parent]["subtopics"]:
            new_tax[matched_parent]["subtopics"][name] = {
                "desc":     ns.get("desc", ""),
                "keywords": [k.lower() for k in ns.get("keywords", [])],
                "source":   "author-seed",
            }
            log.info("  + new sub-topic [%s] > %s", matched_parent, name)
            n_new_subtopics += 1

    log.info("Author seed done — %d new keywords, %d new sub-topics from %d papers.",
             n_new_keywords, n_new_subtopics, len(papers))
    return new_tax


# ---------------------------------------------------------------------------
# PAPER DB  (papers.json)
# ---------------------------------------------------------------------------
def load_paper_db():
    if PAPERS_FILE.exists():
        return json.loads(PAPERS_FILE.read_text())
    return []


def save_paper_db(db):
    PAPERS_FILE.write_text(json.dumps(db, indent=2, ensure_ascii=False))


def add_to_paper_db(db, paper, subtopics, summary_text):
    db.append({
        "id":        paper["base_id"],
        "title":     paper["title"],
        "date":      paper["published"],
        "subtopics": subtopics,
        "abstract":  paper["abstract"][:400],
        "summary":   summary_text[:300],
    })
    # Keep only the most recent 500 entries so the file stays manageable
    if len(db) > 500:
        db[:] = db[-500:]


def few_shot_examples(db, subtopic_names, n=TRIAGE_EXAMPLES):
    """
    Return up to n recent papers whose subtopics overlap with the candidates,
    formatted as triage examples for the prompt.
    """
    relevant = [e for e in reversed(db)
                if any(s in subtopic_names for s in e.get("subtopics", []))]
    others   = [e for e in reversed(db) if e not in relevant]
    chosen   = (relevant + others)[:n]
    if not chosen:
        return ""
    lines = ["Recent classified examples (for context):"]
    for e in chosen:
        lines.append(f'  Title: {e["title"]}')
        lines.append(f'  Sub-topics: {", ".join(e["subtopics"])}')
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. FETCH  (via OAI-PMH — the official arXiv bulk-harvesting interface)
# ---------------------------------------------------------------------------
# Why OAI-PMH instead of the Atom/REST API?
#   • No rate limits on incremental harvests — designed for exactly this use.
#   • Resumption tokens let us page through large result sets cleanly.
#   • The `from` parameter gives precise date-based filtering.
#   • Base URL: https://oaipmh.arxiv.org/oai  (updated March 2025)
#   • Docs:     https://info.arxiv.org/help/oa/index.html
#
# OAI-PMH sets for astro-ph sub-categories use the pattern:
#   physics:astro-ph.SR   (group:archive.SUBCATEGORY)
# All astro-ph: physics:astro-ph   — full astro-ph archive
# ---------------------------------------------------------------------------

def _oai_set_for_category(cat):
    """
    Map an arXiv category string to its OAI-PMH set spec.

    The OAI-PMH set hierarchy is group:archive:SUBCATEGORY where SUBCATEGORY
    is only the suffix code (uppercase), not the full dotted name.  Confirmed
    against https://oaipmh.arxiv.org/oai?verb=ListSets

    Examples:
        "astro-ph.SR"  -> "physics:astro-ph:SR"
        "astro-ph"     -> "physics:astro-ph"
        "cs.AI"        -> "cs:cs:AI"
        "math.NT"      -> "math:math:NT"
        "physics.optics" -> "physics:physics:optics"
    """
    parts = cat.split(".")
    archive = parts[0]          # e.g. "astro-ph", "cs", "math"
    # Determine group from well-known archives
    _GROUP = {
        "astro-ph": "physics", "cond-mat": "physics", "gr-qc": "physics",
        "hep-ex": "physics", "hep-lat": "physics", "hep-ph": "physics",
        "hep-th": "physics", "math-ph": "physics", "nlin": "physics",
        "nucl-ex": "physics", "nucl-th": "physics", "physics": "physics",
        "quant-ph": "physics",
        "cs": "cs", "econ": "econ", "eess": "eess", "math": "math",
        "q-bio": "q-bio", "q-fin": "q-fin", "stat": "stat",
    }
    group = _GROUP.get(archive, archive)
    if len(parts) == 1:
        return f"{group}:{archive}"
    # Subcategory is the suffix only (e.g. "SR" from "astro-ph.SR")
    subcategory = parts[1]
    return f"{group}:{archive}:{subcategory}"


def _oai_parse_records(xml_text):
    """
    Parse OAI-PMH ListRecords XML and return a list of raw record dicts.
    Uses the arXiv native metadata format (metadataPrefix=arXiv).
    """
    import xml.etree.ElementTree as ET

    NS = {
        "oai":   "http://www.openarchives.org/OAI/2.0/",
        "arxiv": "http://arxiv.org/OAI/arXiv/",
    }

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.error("OAI-PMH XML parse error: %s", exc)
        return [], None

    # Check for OAI errors (e.g. noRecordsMatch, badArgument)
    error_el = root.find(".//oai:error", NS)
    if error_el is not None:
        code = error_el.get("code", "unknown")
        msg  = (error_el.text or "").strip()
        if code == "noRecordsMatch":
            log.info("  OAI-PMH: noRecordsMatch (no new papers in date range)")
        else:
            log.error("  OAI-PMH error [%s]: %s", code, msg)
        return [], None

    # Extract resumption token (None = last page)
    token_el = root.find(".//oai:resumptionToken", NS)
    resumption_token = (token_el.text or "").strip() if token_el is not None else None
    if not resumption_token:
        resumption_token = None

    records = []
    for rec in root.findall(".//oai:record", NS):
        # Skip deleted records
        header = rec.find("oai:header", NS)
        if header is not None and header.get("status") == "deleted":
            continue

        meta = rec.find(".//arxiv:arXiv", NS)
        if meta is None:
            continue

        def _text(tag):
            el = meta.find(f"arxiv:{tag}", NS)
            return (el.text or "").strip() if el is not None else ""

        arxiv_id = _text("id")           # e.g. "2501.12345"
        title    = re.sub(r"\s+", " ", _text("title").replace("\n", " ")).strip()
        abstract = re.sub(r"\s+", " ", _text("abstract").replace("\n", " ")).strip()

        # Authors
        authors = []
        for author_el in meta.findall("arxiv:authors/arxiv:author", NS):
            forenames = (author_el.findtext("arxiv:forenames", "", NS) or "").strip()
            keyname   = (author_el.findtext("arxiv:keyname",   "", NS) or "").strip()
            name = f"{forenames} {keyname}".strip() if forenames else keyname
            if name:
                authors.append(name)

        # Categories (primary first, then cross-listed)
        primary_cat = _text("categories").split()[0] if _text("categories") else ""
        categories  = _text("categories").split()

        # Submission date — use the datestamp from the OAI header as proxy.
        # arXiv's created/updated fields are also in the arXiv namespace.
        created_str = _text("created")   # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
        if created_str:
            date_part = created_str[:10]
        else:
            # Fall back to header datestamp
            ds = header.findtext("oai:datestamp", "", NS) if header is not None else ""
            date_part = ds[:10] if ds else ""

        if not arxiv_id or not title:
            continue

        base_id = re.sub(r"v\d+$", "", arxiv_id)
        records.append({
            "arxiv_id":  arxiv_id,
            "base_id":   base_id,
            "title":     title,
            "abstract":  abstract,
            "authors":   authors,
            "date_str":  date_part,         # "YYYY-MM-DD"
            "categories": categories,
            "primary":   primary_cat,
        })

    return records, resumption_token


def fetch_feed(lookback_days, save_feed=False):
    """
    Harvest recent papers from arXiv using OAI-PMH (no rate limits).

    Strategy:
      • Build a `from` date = today − lookback_days.
      • Request one set per CATEGORIES entry (or all in one go if they share
        the same archive — the typical case for astro-ph.SR).
      • Follow resumption tokens to page through results.
      • Filter to papers whose creation date falls inside the window.
    """
    import xml.etree.ElementTree as ET

    cutoff   = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    from_str = cutoff.strftime("%Y-%m-%d")

    # Build the set list — one OAI set per requested category.
    # If multiple categories share one archive (e.g. astro-ph.SR + astro-ph.GA)
    # we could harvest the parent set once; for simplicity we harvest each set.
    sets = [_oai_set_for_category(cat) for cat in CATEGORIES]
    log.info("STAGE 1 fetch — OAI-PMH harvest | sets=%s | from=%s | max=%d",
             sets, from_str, MAX_FETCH)

    all_raw   = []    # raw record dicts from XML
    seen_ids  = set() # dedup across sets

    for oai_set in sets:
        params = {
            "verb":            "ListRecords",
            "metadataPrefix":  "arXiv",
            "set":             oai_set,
            "from":            from_str,
        }
        page = 0
        while True:
            url = OAI_PMH_URL + "?" + requests.compat.urlencode(params)
            log.debug("OAI-PMH GET %s", url)
            try:
                resp = SESSION.get(url, timeout=HTTP_TIMEOUT)
            except requests.RequestException as exc:
                log.error("OAI-PMH request failed: %s", exc)
                break

            if resp.status_code != 200:
                log.error("OAI-PMH HTTP %s — aborting set %s", resp.status_code, oai_set)
                break

            if save_feed:
                fname = HERE / f"feed_debug_p{page}.xml"
                fname.write_text(resp.text, encoding="utf-8")
                log.info("raw OAI page saved to %s", fname)

            records, token = _oai_parse_records(resp.text)
            log.debug("  page %d: %d records, token=%s", page, len(records),
                      bool(token))

            for r in records:
                if r["base_id"] not in seen_ids:
                    seen_ids.add(r["base_id"])
                    all_raw.append(r)

            # Stop if we already have enough or there are no more pages
            if token is None or len(all_raw) >= MAX_FETCH:
                break

            # Follow resumption token
            params = {"verb": "ListRecords", "resumptionToken": token}
            page  += 1
            time.sleep(OAI_PAGE_DELAY)

    log.info("  OAI-PMH returned %d unique records (before date/category filter)",
             len(all_raw))

    if not all_raw:
        log.error("  No records returned — check OAI sets or widen --days.")
        return []

    # Convert to the paper dict format used by the rest of the pipeline.
    # Filter: keep only papers whose primary category is in CATEGORIES and
    # whose creation date falls within the lookback window.
    cat_set = set(CATEGORIES)
    papers, too_old, wrong_cat = [], 0, 0
    for r in all_raw:
        # Category filter (primary category must be requested)
        if not any(c in cat_set for c in r["categories"]):
            wrong_cat += 1
            continue

        # Date filter
        if r["date_str"]:
            try:
                pub_date = datetime.strptime(r["date_str"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc)
            except ValueError:
                pub_date = None
        else:
            pub_date = None

        if pub_date and pub_date < cutoff:
            too_old += 1
            continue

        base_id = r["base_id"]
        papers.append({
            "id":        r["arxiv_id"],
            "base_id":   base_id,
            "title":     r["title"],
            "abstract":  r["abstract"],
            "authors":   r["authors"],
            "published": r["date_str"] or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "categories": r["categories"],
            "abs_url":   f"https://arxiv.org/abs/{base_id}",
            "pdf_url":   f"https://arxiv.org/pdf/{base_id}",
        })

        if len(papers) >= MAX_FETCH:
            break

    log.info("  %d papers kept (lookback=%dd); %d too old, %d wrong category",
             len(papers), lookback_days, too_old, wrong_cat)

    if papers:
        log.info("  newest: %s  oldest: %s",
                 max(p["published"] for p in papers),
                 min(p["published"] for p in papers))
    else:
        log.warning("  No papers in window — try --days %d.", lookback_days + 4)

    return papers


# ---------------------------------------------------------------------------
# 2. KEYWORD PRE-FILTER  (uses full flattened taxonomy)
# ---------------------------------------------------------------------------
def load_seen():
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()


def save_seen(seen):
    SEEN_FILE.write_text(json.dumps(sorted(seen), indent=0))


def keyword_match(paper, tax, active_subtopics=None):
    """
    Returns list of matching subtopic names.
    active_subtopics: if set, restricts to that subset (from --topics flag).
    """
    hay  = (paper["title"] + " " + paper["abstract"]).lower()
    flat = all_subtopics(tax)
    hits = []
    for sname, s in flat.items():
        if active_subtopics and sname not in active_subtopics:
            continue
        if any(kw in hay for kw in s.get("keywords", [])):
            hits.append(sname)
    return hits


# ---------------------------------------------------------------------------
# 3. OLLAMA HELPERS
# ---------------------------------------------------------------------------
def check_ollama():
    try:
        r = SESSION.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        names = [m["name"] for m in r.json().get("models", [])]
        log.debug("Ollama models: %s", names)
        if OLLAMA_MODEL not in names and not any(OLLAMA_MODEL in n for n in names):
            log.warning("Ollama up but '%s' not found in: %s", OLLAMA_MODEL, names)
        else:
            log.info("Ollama reachable; model '%s' present", OLLAMA_MODEL)
        return True
    except requests.RequestException as e:
        log.error("Cannot reach Ollama at %s (%s). Is `ollama serve` running?", OLLAMA_URL, e)
        return False


def ollama(prompt, json_mode=False, num_predict=512):
    body = {
        "model":  OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"num_ctx": NUM_CTX, "num_predict": num_predict, "temperature": 0.2},
    }
    if json_mode:
        body["format"] = "json"
    t0 = time.time()
    r  = SESSION.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=300)
    r.raise_for_status()
    out = r.json()["response"].strip()
    log.debug("  ollama %.1fs, %d chars out", time.time() - t0, len(out))
    return out


# ---------------------------------------------------------------------------
# 3b. EMBEDDINGS  (semantic similarity triage)
# ---------------------------------------------------------------------------
def _dot(a, b):
    """Cosine similarity for two plain Python float lists."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _embed_model():
    """Return the embedding model name to use, falling back to OLLAMA_MODEL."""
    try:
        r = SESSION.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        names = [m["name"] for m in r.json().get("models", [])]
        if any(EMBED_MODEL in n for n in names):
            return EMBED_MODEL
        log.debug("  %s not found in Ollama — falling back to %s for embeddings",
                  EMBED_MODEL, OLLAMA_MODEL)
    except Exception:
        pass
    return OLLAMA_MODEL


def embed(texts, model=None):
    """
    Embed a list of strings via Ollama /api/embed.
    Returns a list of float vectors, one per input text.
    """
    if model is None:
        model = _embed_model()
    body = {"model": model, "input": texts}
    t0   = time.time()
    r    = SESSION.post(f"{OLLAMA_URL}/api/embed", json=body, timeout=120)
    r.raise_for_status()
    vecs = r.json()["embeddings"]
    log.debug("  embed %d text(s) via %s in %.1fs", len(texts), model, time.time() - t0)
    return vecs


def _taxonomy_fingerprint(tax):
    """A cheap hash of the taxonomy so we know when to rebuild embeddings."""
    import hashlib
    flat = all_subtopics(tax)
    blob = json.dumps(
        {s: {"desc": v["desc"], "keywords": sorted(v.get("keywords", []))}
         for s, v in sorted(flat.items())},
        sort_keys=True
    )
    return hashlib.md5(blob.encode()).hexdigest()


def load_subtopic_embeddings(tax):
    """
    Load cached sub-topic embeddings from disk, rebuilding if taxonomy changed.
    Returns dict: subtopic_name -> embedding vector.
    """
    fingerprint = _taxonomy_fingerprint(tax)
    if EMBED_FILE.exists():
        cached = json.loads(EMBED_FILE.read_text())
        if cached.get("fingerprint") == fingerprint and "background" in cached:
            log.debug("Sub-topic embeddings loaded from cache (%d entries)",
                      len(cached["vectors"]))
            return cached["vectors"], cached["background"]
        log.info("Taxonomy changed or background missing — rebuilding embeddings …")
    else:
        log.info("Building sub-topic embeddings for the first time …")

    flat   = all_subtopics(tax)
    model  = _embed_model()
    # Each sub-topic is represented by: description + all keywords joined
    texts  = []
    names  = []
    for sname, s in flat.items():
        kws  = ", ".join(s.get("keywords", []))
        texts.append(f"{sname}. {s['desc']} Keywords: {kws}")
        names.append(sname)

    # Embed sub-topics and background in one batch call
    all_texts = texts + [EMBED_BACKGROUND]
    all_vecs  = embed(all_texts, model=model)
    vectors   = {name: vec for name, vec in zip(names, all_vecs)}
    bg_vec    = all_vecs[-1]
    EMBED_FILE.write_text(json.dumps(
        {"fingerprint": fingerprint, "model": model,
         "vectors": vectors, "background": bg_vec},
        indent=None  # compact — these are large float arrays
    ))
    log.info("  %d sub-topic vectors + background cached to %s",
             len(vectors), EMBED_FILE.name)
    return vectors, bg_vec


def embed_triage(paper, subtopic_vecs, bg_vec, active_subtopics=None):
    """
    Embed paper and score *lift* = cosine(paper, subtopic) - cosine(paper, background).
    Lift isolates genuine topic relevance from generic astro-ph.SR similarity.

    Returns:
        verdict   : 'accept' | 'review' | 'reject'
        subtopics : list of matched sub-topic names (non-empty for accept/review)
        scores    : dict subtopic -> lift score (for debug logging)
    """
    text = f"{paper['title']}. {paper['abstract']}"
    try:
        paper_vec = embed([text])[0]
    except Exception as e:
        log.warning("  embed failed (%s) — falling back to Ollama triage", e)
        return "review", [], {}

    bg_score = _dot(paper_vec, bg_vec)
    lifts = {}
    for sname, svec in subtopic_vecs.items():
        if active_subtopics and sname not in active_subtopics:
            continue
        lifts[sname] = _dot(paper_vec, svec) - bg_score

    if not lifts:
        return "reject", [], lifts

    best_lift = max(lifts.values())
    matched   = sorted(
        [s for s, sc in lifts.items() if sc >= EMBED_REVIEW],
        key=lambda s: lifts[s], reverse=True
    )

    if best_lift >= EMBED_ACCEPT:
        verdict = "accept"
    elif best_lift >= EMBED_REVIEW:
        verdict = "review"
    else:
        verdict = "reject"

    log.debug("  embed_triage bg=%.3f best_lift=%.3f verdict=%s matched=%s",
              bg_score, best_lift, verdict, matched[:3])
    return verdict, matched, lifts


# ---------------------------------------------------------------------------
# 4. TRIAGE  (batch sub-topic assignment — one Ollama call per N papers)
# ---------------------------------------------------------------------------
def batch_triage(papers_hits, tax, paper_db):
    """
    Triage a batch of (paper, candidate_subtopics) pairs in a single Ollama call.
    Returns list of (relevant: bool, subtopics: list) in the same order.

    One call for 10 papers takes ~35s vs ~300s for 10 individual calls.
    """
    flat = all_subtopics(tax)

    # Build a compact sub-topic reference (shared across all papers in batch)
    all_candidate_subs = sorted({s for _, hits in papers_hits for s in hits})
    subtopic_ref = "\n".join(
        f'  {i+1}. "{s}": {flat[s]["desc"]}'
        for i, s in enumerate(all_candidate_subs) if s in flat
    )

    # Few-shot examples from paper DB
    examples = few_shot_examples(paper_db, all_candidate_subs)

    # Build the per-paper block
    paper_blocks = []
    for idx, (p, hits) in enumerate(papers_hits):
        candidate_nums = [str(all_candidate_subs.index(s)+1)
                          for s in hits if s in all_candidate_subs]
        paper_blocks.append(
            f'Paper {idx+1} (candidates: {", ".join(candidate_nums)}):\n'
            f'  Title: {p["title"]}\n'
            f'  Abstract: {p["abstract"][:800]}'
        )

    prompt = (
        "You are screening astrophysics preprints for a researcher studying "
        "eclipsing binaries and asteroseismology.\n\n"
        f"Sub-topic reference:\n{subtopic_ref}\n\n"
        + (f"{examples}\n\n" if examples else "")
        + "Papers to screen:\n\n"
        + "\n\n".join(paper_blocks)
        + "\n\nFor each paper, decide if it is substantially about any of its "
          "candidate sub-topics.\n"
          "Respond with a JSON array, one entry per paper, in order:\n"
          '[\n'
          '  {"paper": 1, "relevant": true/false, "subtopics": ["<name>", ...], "reason": "<clause>"},\n'
          '  ...\n'
          ']'
    )
    try:
        raw     = ollama(prompt, json_mode=True,
                         num_predict=80 * len(papers_hits))
        # The model may wrap the array in an object; unwrap if needed
        parsed  = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = next((v for v in parsed.values()
                           if isinstance(v, list)), [])
        results = []
        for i, (p, hits) in enumerate(papers_hits):
            entry     = next((e for e in parsed
                              if isinstance(e, dict) and e.get("paper") == i + 1),
                             None)
            if entry is None:
                log.warning("  batch_triage: no result for paper %d — keeping", i+1)
                results.append((True, hits[:1]))
                continue
            subs = entry.get("subtopics") or []
            if isinstance(subs, str):
                subs = [subs]
            if entry.get("relevant") and not subs:
                subs = hits[:1]
            results.append((bool(entry.get("relevant")), subs))
            log.debug("  batch paper %d: relevant=%s subtopics=%s reason=%s",
                      i+1, entry.get("relevant"), subs, entry.get("reason",""))
        return results
    except Exception as e:
        log.warning("  batch_triage error (%s) — falling back to accept-all", e)
        return [(True, hits[:1]) for _, hits in papers_hits]


# ---------------------------------------------------------------------------
# 5. FETCH FULL TEXT
# ---------------------------------------------------------------------------
def _extract_sections_from_html(html_text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text, "html.parser")
    abstract = ""
    a = soup.find(class_=re.compile("ltx_abstract"))
    if a:
        abstract = " ".join(a.get_text(" ", strip=True).split())
    intro, methods, concl = "", "", ""
    re_intro   = re.compile(r"\bintroduction\b", re.I)
    re_methods = re.compile(r"\b(method|observation|data|instrument|reduction)\b", re.I)
    re_concl   = re.compile(r"\b(conclusion|conclusions|summary|concluding)\b", re.I)
    for sec in soup.find_all(class_=re.compile("ltx_section")):
        head = sec.find(class_=re.compile("ltx_title"))
        if not head:
            continue
        htext = head.get_text(" ", strip=True)
        body  = " ".join(sec.get_text(" ", strip=True).split())
        if not intro and re_intro.search(htext):
            intro = body
        elif not methods and re_methods.search(htext):
            methods = body
        elif re_concl.search(htext):
            concl = body
    return abstract, intro, methods, concl


_FULLTEXT_MIN_DELAY = 3.0  # polite delay between HTML/PDF fulltext fetches

def _arxiv_get(url, **kw):
    # Rate-limited GET: enforces _FULLTEXT_MIN_DELAY globally across threads
    # for HTML/PDF fulltext fetches.  Concurrent callers queue here rather
    # than each sleeping independently.
    with _ARXIV_FETCH_LOCK:
        elapsed = time.time() - _ARXIV_LAST_REQ[0]
        gap = _FULLTEXT_MIN_DELAY - elapsed
        if gap > 0:
            time.sleep(gap)
        r = SESSION.get(url, **kw)
        _ARXIV_LAST_REQ[0] = time.time()
    return r


def fetch_fulltext(paper):
    base = paper["base_id"]
    for label, url in (("arxiv-html", f"https://arxiv.org/html/{base}"),
                       ("ar5iv",      f"https://ar5iv.org/abs/{base}")):
        try:
            r = _arxiv_get(url, timeout=30)
            log.debug("  %s -> HTTP %s", label, r.status_code)
            if "rate exceeded" in r.text[:200].lower():
                log.warning("  rate-limit on %s — skipping", label)
                time.sleep(15)
                continue
            if r.status_code == 200 and "ltx_" in r.text:
                _, intro, methods, concl = _extract_sections_from_html(r.text)
                if intro or concl:
                    log.debug("  extracted intro=%d methods=%d concl=%d chars via %s",
                              len(intro), len(methods), len(concl), label)
                    return intro, methods, concl, label
                log.debug("  %s had no recognisable sections", label)
        except requests.RequestException as e:
            log.debug("  %s failed: %s", label, e)
    try:
        import fitz
        r    = _arxiv_get(paper["pdf_url"], timeout=60)
        data = r.content
        del r                                      # release HTTP response buffer
        with fitz.open(stream=data, filetype="pdf") as doc:
            del data                               # fitz has its own copy; free ours
            n    = doc.page_count
            head = "\n".join(doc[i].get_text() for i in range(min(3, n)))
            tail = "\n".join(doc[i].get_text() for i in range(max(0, n - 3), n))
        log.debug("  PDF fallback (%d pages)", n)
        return head, "", tail, "pdf"
    except Exception as e:
        log.debug("  PDF fallback unavailable (%s); abstract-only", e)
        return "", "", "", "abstract-only"


# ---------------------------------------------------------------------------
# 6. SUMMARISE  (longer, includes methods paragraph)
# ---------------------------------------------------------------------------
def summarize(paper, intro, methods, concl, subtopics):
    context = (f"TITLE: {paper['title']}\n\n"
               f"ABSTRACT: {paper['abstract'][:ABSTRACT_CHARS]}\n\n")
    if intro:
        context += f"INTRODUCTION (excerpt): {intro[:INTRO_CHARS]}\n\n"
    if methods:
        context += f"METHODS/OBSERVATIONS (excerpt): {methods[:METHODS_CHARS]}\n\n"
    if concl:
        context += f"CONCLUSIONS (excerpt): {concl[:CONCLUSION_CHARS]}\n\n"

    subtopic_str = ", ".join(subtopics) if subtopics else "eclipsing binaries / asteroseismology"
    prompt = (
        "You are an expert stellar astrophysicist writing a digest entry for a "
        "researcher specialising in eclipsing binaries and asteroseismology. "
        f"This paper is classified under: {subtopic_str}.\n\n"
        f"{context}"
        "Write a detailed but focused digest entry. "
        "Respond with JSON only, no prose outside it:\n"
        "{\n"
        '  "summary": "<4-6 sentences: scientific question, approach, and main result>",\n'
        '  "methods": "<3-4 sentences: instruments/surveys, codes, statistical approach, '
        'key parameters fitted or derived>",\n'
        '  "key_findings": ["<specific quantitative finding>", "<finding>", "<finding>"],\n'
        '  "conclusions": "<2-3 sentences: broader implications and takeaway>",\n'
        '  "relevance": "<one sentence: why this matters for EB/asteroseismology research>"\n'
        "}"
    )
    try:
        out = json.loads(ollama(prompt, json_mode=True, num_predict=1200))
        if isinstance(out.get("key_findings"), str):
            out["key_findings"] = [out["key_findings"]]
        return out
    except Exception as e:
        log.warning("  summarize error (%s); using abstract as fallback", e)
        return {"summary": paper["abstract"], "methods": "", "key_findings": [],
                "conclusions": "", "relevance": ""}


# ---------------------------------------------------------------------------
# 7. TAXONOMY EXPANSION  (--expand-taxonomy)
# ---------------------------------------------------------------------------
def expand_taxonomy(tax, lookback_days=14):
    """
    Fetch recent astro-ph.SR papers, ask Qwen to suggest new sub-topics or
    keywords it notices that aren't already covered, then merge into taxonomy.
    """
    log.info("EXPAND TAXONOMY — fetching recent papers for Qwen to analyse …")
    papers = fetch_feed(lookback_days)
    if not papers:
        log.warning("No papers fetched — cannot expand taxonomy.")
        return tax

    # Use all keyword hits (no active filter) to find a broad sample
    sample = []
    kws = all_keywords(tax)
    for p in papers:
        hay = (p["title"] + " " + p["abstract"]).lower()
        if any(kw in hay for kw in kws):
            sample.append(p)
        if len(sample) >= 30:
            break

    if not sample:
        log.warning("No keyword-matching papers in sample for taxonomy expansion.")
        return tax

    log.info("  Showing Qwen %d papers for taxonomy expansion …", len(sample))

    # Summarise current taxonomy for the prompt
    tax_summary = []
    for tname, t in tax.items():
        tax_summary.append(f"Topic: {tname}")
        for sname in t.get("subtopics", {}):
            tax_summary.append(f"  Sub-topic: {sname}")
    tax_str = "\n".join(tax_summary)

    paper_block = "\n\n".join(
        f"Title: {p['title']}\nAbstract: {p['abstract'][:600]}"
        for p in sample
    )

    prompt = (
        "You are an expert in stellar astrophysics, eclipsing binaries, and asteroseismology.\n\n"
        "Current taxonomy:\n"
        f"{tax_str}\n\n"
        "Below are recent arXiv papers in astro-ph.SR. Study them and propose:\n"
        "1. New sub-topics not yet in the taxonomy that appear repeatedly.\n"
        "2. New keywords for existing sub-topics that would improve recall.\n\n"
        f"{paper_block}\n\n"
        "Respond with JSON only:\n"
        "{\n"
        '  "new_subtopics": [\n'
        '    {"parent_topic": "<existing topic name>", "name": "<sub-topic name>",\n'
        '     "desc": "<one sentence>", "keywords": ["kw1", "kw2", ...]},\n'
        '    ...\n'
        '  ],\n'
        '  "keyword_additions": [\n'
        '    {"subtopic": "<existing sub-topic name>", "keywords": ["kw1", ...]},\n'
        '    ...\n'
        '  ]\n'
        '}'
    )

    try:
        raw = ollama(prompt, json_mode=True, num_predict=1500)
        proposals = json.loads(raw)
    except Exception as e:
        log.error("Qwen taxonomy expansion failed: %s", e)
        return tax

    new_tax = copy.deepcopy(tax)
    n_new_subtopics = 0
    n_new_keywords  = 0

    for ns in proposals.get("new_subtopics", []):
        parent = ns.get("parent_topic", "")
        name   = ns.get("name", "").strip()
        if not name or not parent:
            continue
        # Find parent case-insensitively
        matched_parent = next(
            (k for k in new_tax if k.lower() == parent.lower()), None)
        if matched_parent is None:
            log.debug("  expansion: unknown parent topic %r — skipping", parent)
            continue
        if name not in new_tax[matched_parent]["subtopics"]:
            new_tax[matched_parent]["subtopics"][name] = {
                "desc":     ns.get("desc", ""),
                "keywords": [k.lower() for k in ns.get("keywords", [])],
                "source":   "qwen-expansion",
            }
            log.info("  + new sub-topic: [%s] %s", matched_parent, name)
            n_new_subtopics += 1

    for ka in proposals.get("keyword_additions", []):
        subtopic = ka.get("subtopic", "").strip()
        new_kws  = [k.lower() for k in ka.get("keywords", [])]
        for tname, t in new_tax.items():
            if subtopic in t.get("subtopics", {}):
                existing = set(t["subtopics"][subtopic].get("keywords", []))
                added    = [k for k in new_kws if k not in existing]
                t["subtopics"][subtopic].setdefault("keywords", []).extend(added)
                if added:
                    log.info("  + keywords for [%s]: %s", subtopic, added)
                    n_new_keywords += len(added)
                break

    log.info("Taxonomy expansion done: %d new sub-topics, %d new keywords.",
             n_new_subtopics, n_new_keywords)
    return new_tax


def _render_keyword_appendix(all_papers, tax):
    """
    Build an HTML section listing every fetched paper whose title or abstract
    contains at least one taxonomy keyword, grouped by the subtopics that matched.
    Papers already fully summarised in the main cards are included too — they
    appear here as a quick-reference index alongside the ones that were filtered
    out by embedding/triage.
    """
    flat = all_subtopics(tax)

    # For each paper collect which subtopics matched and which keywords triggered
    rows = []   # (paper_dict, [(subtopic_name, [matched_kws]), ...])
    for p in all_papers:
        hay  = (p["title"] + " " + p["abstract"]).lower()
        hits = {}   # subtopic_name -> list of matched keywords
        for sname, s in flat.items():
            matched_kws = [kw for kw in s.get("keywords", []) if kw in hay]
            if matched_kws:
                hits[sname] = matched_kws
        if hits:
            rows.append((p, hits))

    if not rows:
        return ""

    # Group rows by their top-level parent topic
    by_parent = {}
    for p, hits in rows:
        # A paper may span multiple parents; list it under each
        parents = {flat[sname]["parent"] for sname in hits}
        for parent in parents:
            by_parent.setdefault(parent, []).append((p, hits))

    sections = []
    for parent, entries in by_parent.items():
        # De-duplicate papers within this parent (same paper may hit multiple subtopics)
        seen_ids = set()
        li_items = []
        for p, hits in entries:
            if p["base_id"] in seen_ids:
                continue
            seen_ids.add(p["base_id"])

            # Collect only the subtopics that belong to this parent
            parent_hits = {
                sname: kws for sname, kws in hits.items()
                if flat[sname]["parent"] == parent
            }
            subtopic_badges = " ".join(
                f"<span class='kw-sub'>{_esc(sname)}</span>"
                for sname in parent_hits
            )
            # Show a sample of the triggering keywords (cap at 5 per paper)
            all_kws = sorted({kw for kws in parent_hits.values() for kw in kws})
            kw_str  = ", ".join(all_kws[:5]) + ("…" if len(all_kws) > 5 else "")

            li_items.append(
                f"<li class='kw-item'>"
                f"<a href='https://arxiv.org/abs/{_esc(p['base_id'])}' target='_blank'>"
                f"{_esc(p['title'])}</a> "
                f"<span class='kw-date'>{_esc(p['published'])}</span> "
                f"{subtopic_badges}"
                f"<span class='kw-triggers'>{_esc(kw_str)}</span>"
                f"</li>"
            )

        if li_items:
            sections.append(
                f"<div class='kw-group'>"
                f"<h3 class='kw-heading'>{_esc(parent)}</h3>"
                f"<ul class='kw-list'>{''.join(li_items)}</ul>"
                f"</div>"
            )

    toggle_id = "kw-appendix-body"
    return f"""
<section class='kw-appendix' id='kw-appendix'>
  <h2 class='kw-appendix-title'>
    Keyword-matched papers
    <button class='kw-toggle' onclick="
      var b=document.getElementById('{toggle_id}');
      var btn=this;
      if(b.style.display==='none'){{b.style.display='block';btn.textContent='▲ collapse';}}
      else{{b.style.display='none';btn.textContent='▼ expand';}}
    ">▼ expand</button>
    <span class='kw-count'>{len(rows)} paper(s)</span>
  </h2>
  <div id='{toggle_id}' style='display:none'>
    {''.join(sections)}
  </div>
</section>"""

# ---------------------------------------------------------------------------
# 8. RENDER
# ---------------------------------------------------------------------------
def _esc(s):
    return html.escape(str(s)) if s else ""


def render_report(date_str, items, tax):
    cards = []
    for it in items:
        p, s   = it["paper"], it["summary"]
        parent = it.get("parent_topic", "")
        subs   = it.get("subtopics", [])

        findings = "".join(f"<li>{_esc(f)}</li>" for f in s.get("key_findings", []) if f)
        findings_html = f"<ul class='findings'>{findings}</ul>" if findings else ""
        authors = ", ".join(p["authors"][:6]) + (" et al." if len(p["authors"]) > 6 else "")

        sub_badges = " ".join(
            f"<span class='subtopic'>{_esc(sub)}</span>" for sub in subs)
        meta = [
            f"<span class='topic'>{_esc(parent)}</span>",
            sub_badges,
            f"<span>{_esc(p['published'])}</span>",
        ]
        cards.append(f"""
        <article class="card">
          <h2><a href="{_esc(p['abs_url'])}" target="_blank">{_esc(p['title'])}</a></h2>
          <p class="authors">{_esc(authors)}</p>
          <p class="meta">{' &middot; '.join(m for m in meta if m)}</p>
          <p class="summary">{_esc(s.get('summary'))}</p>
          {f"<p class='methods'><b>Methods.</b> {_esc(s.get('methods'))}</p>" if s.get('methods') else ''}
          {findings_html}
          {f"<p class='concl'><b>Conclusions.</b> {_esc(s.get('conclusions'))}</p>" if s.get('conclusions') else ''}
          {f"<p class='aux'><b>Why it matters.</b> {_esc(s.get('relevance'))}</p>" if s.get('relevance') else ''}
          <p class="links">
            <a href="{_esc(p['abs_url'])}" target="_blank">abstract</a>
            &middot; <a href="{_esc(p['pdf_url'])}" target="_blank">pdf</a>
          </p>
        </article>""")

    all_topics = ", ".join(tax.keys())
    body = "\n".join(cards) if cards else "<p class='empty'>No matching papers today.</p>"

    # Keyword appendix — appended after the main cards
    appendix = _render_keyword_appendix(all_papers, tax) if all_papers else ""

    page = REPORT_TEMPLATE.format(
        date=date_str, count=len(items), body=body + appendix,   # ← append here
        model=_esc(OLLAMA_MODEL), topics=_esc(all_topics))
    out = REPORT_DIR / f"{date_str}.html"
    out.write_text(page, encoding="utf-8")
    _rebuild_index()
    return out


def _rebuild_index():
    reports = sorted(REPORT_DIR.glob("20*.html"), reverse=True)
    rows = "".join(f"<li><a href='{r.name}'>{r.stem}</a></li>" for r in reports)
    (REPORT_DIR / "index.html").write_text(
        INDEX_TEMPLATE.format(rows=rows or "<li>No reports yet.</li>"), encoding="utf-8")


# ---------------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------------
def calibrate_thresholds(tax, subtopic_vecs, bg_vec, lookback_days=7, active_subtopics=None):
    """
    Fetch a sample of recent papers, embed them, and print a score distribution
    so you can see where relevant and irrelevant papers actually cluster.
    Suggests EMBED_ACCEPT and EMBED_REVIEW values based on the gap.
    """
    print("\nFetching recent papers for calibration …")
    papers = fetch_feed(lookback_days)
    if not papers:
        print("No papers fetched.")
        return

    # Score every paper that passes keyword filter
    flat = all_subtopics(tax)
    rows = []   # (base_id, title, best_score, best_subtopic, kw_hit)
    for p in papers:
        kw_hits = keyword_match(p, tax, active_subtopics)
        text    = f"{p['title']}. {p['abstract']}"
        try:
            vec = embed([text])[0]
        except Exception as e:
            log.warning("embed failed for %s: %s", p["base_id"], e)
            continue

        bg   = _dot(vec, bg_vec)
        lifts = {}
        for sname, svec in subtopic_vecs.items():
            if active_subtopics and sname not in active_subtopics:
                continue
            lifts[sname] = _dot(vec, svec) - bg

        if not lifts:
            continue

        best_sub   = max(lifts, key=lifts.__getitem__)
        best_score = lifts[best_sub]
        rows.append((p["base_id"], p["title"], best_score, best_sub, bool(kw_hits)))

    if not rows:
        print("No papers could be scored.")
        return

    rows.sort(key=lambda r: r[2], reverse=True)
    all_scores  = [r[2] for r in rows]
    kw_scores   = [r[2] for r in rows if r[4]]       # keyword-matched papers
    nokw_scores = [r[2] for r in rows if not r[4]]   # keyword-missed papers

    # ── Print score histogram (buckets of 0.05) ──────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Lift score distribution — {len(rows)} papers, lookback={lookback_days}d")
    print(f"  {'[kw=keyword match]':40s}  {'score':>6}")
    print(f"{'='*65}")
    buckets = {}
    for r in rows:
        b = round(r[2] * 20) / 20   # round to nearest 0.05
        buckets.setdefault(b, []).append(r)
    for b in sorted(buckets, reverse=True):
        grp   = buckets[b]
        bar   = "█" * len(grp)
        kw    = sum(1 for r in grp if r[4])
        label = f"{b:.2f}"
        print(f"  {label}  {bar:<40s}  {len(grp):3d} papers  ({kw} kw-hit)")

    # ── Show top-20 scored papers so you can judge relevance by eye ──────────
    print(f"\n{'='*65}")
    print("  Top 25 papers by best sub-topic score:")
    print(f"{'='*65}")
    for base_id, title, score, sub, kw in rows[:25]:
        kw_tag = "[kw]" if kw else "    "
        print(f"  {score:.3f}  {kw_tag}  {sub[:28]:<28s}  {title[:55]}")

    # ── Show bottom-10 so you see what gets rejected ──────────────────────────
    print(f"\n  Bottom 10 (likely irrelevant):")
    for base_id, title, score, sub, kw in rows[-10:]:
        kw_tag = "[kw]" if kw else "    "
        print(f"  {score:.3f}  {kw_tag}  {sub[:28]:<28s}  {title[:55]}")

    # ── Suggest thresholds ────────────────────────────────────────────────────
    # Strategy: find the largest gap in the sorted score list among the top half.
    # The gap between relevant (high) and noise (low) is usually the biggest jump.
    if len(all_scores) >= 4:
        sorted_scores = sorted(all_scores, reverse=True)
        n_top         = max(4, len(sorted_scores) // 3)   # look in top third
        gaps = [(sorted_scores[i] - sorted_scores[i+1], i, sorted_scores[i+1])
                for i in range(n_top - 1)]
        gaps.sort(reverse=True)
        gap_size, gap_idx, gap_lower = gaps[0]

        # ACCEPT = just above the biggest gap; REVIEW = 0.10 below that
        suggested_accept = round(gap_lower + gap_size * 0.6, 2)
        suggested_review = round(max(0.10, suggested_accept - 0.12), 2)

        print(f"\n{'='*65}")
        print(f"  Largest score gap: {gap_size:.3f} between rank {gap_idx} and {gap_idx+1}")
        print(f"  Suggested thresholds (edit CONFIG at top of script):")
        print(f"    EMBED_ACCEPT = {suggested_accept}   # confident hit — skip Ollama triage")
        print(f"    EMBED_REVIEW = {suggested_review}   # borderline  — send to Ollama triage")
        print()
        if kw_scores and nokw_scores:
            avg_kw   = sum(kw_scores)   / len(kw_scores)
            avg_nokw = sum(nokw_scores) / len(nokw_scores)
            print(f"  Avg score of kw-matched papers : {avg_kw:.3f}")
            print(f"  Avg score of kw-missed papers  : {avg_nokw:.3f}")
            print(f"  Separation                     : {avg_kw - avg_nokw:.3f}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Local arXiv digest — EBs & asteroseismology with living taxonomy")
    ap.add_argument("-v", "--verbose",    action="store_true", help="DEBUG logging")
    ap.add_argument("--dry-run",          action="store_true", help="fetch+filter only, no LLM")
    ap.add_argument("--no-triage",        action="store_true", help="skip LLM triage")
    ap.add_argument("--days",    type=int, default=LOOKBACK_DAYS, help="lookback window")
    ap.add_argument("--reset-seen",       action="store_true", help="forget seen papers")
    ap.add_argument("--limit",   type=int, default=0, help="cap papers summarised")
    ap.add_argument("--save-feed",        action="store_true", help="dump raw feed XML")
    ap.add_argument("--topics",  nargs="+", metavar="SUBTOPIC",
                    help="restrict to topic(s) or sub-topic(s) by name. "
                         "A top-level topic expands to all its sub-topics. "
                         "Use --list-topics to see all names.")
    ap.add_argument("--seed-author", metavar="NAME",
                help="harvest this author's arXiv papers and seed the taxonomy "
                     "with keywords from their work (use --days to set lookback, "
                     "default 365)")
    ap.add_argument("--list-topics",      action="store_true",
                    help="print taxonomy and exit")
    ap.add_argument("--expand-taxonomy",  action="store_true",
                    help="ask Qwen to grow the taxonomy from recent papers, then exit")
    ap.add_argument("--calibrate",        action="store_true",
                    help="score a sample of recent papers and suggest EMBED_ACCEPT / "
                         "EMBED_REVIEW thresholds based on the actual distribution")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    tax = load_taxonomy()

    # -- list-topics --
    if args.list_topics:
        for tname, t in tax.items():
            print(f"\n{'='*60}")
            print(f"TOPIC: {tname}")
            print(f"  {t['desc']}")
            for sname, s in t.get("subtopics", {}).items():
                src = " [qwen]" if s.get("source") == "qwen-expansion" else ""
                print(f"\n  SUB-TOPIC: {sname}{src}")
                print(f"    {s['desc']}")
                print(f"    keywords: {', '.join(s.get('keywords', []))}")
        return

    # -- expand-taxonomy --
    if args.expand_taxonomy:
        if not check_ollama():
            log.error("Ollama unreachable — cannot expand taxonomy.")
            return
        new_tax = expand_taxonomy(tax, lookback_days=args.days or 14)
        save_taxonomy(new_tax)
        log.info("taxonomy.json updated.")
        return

    if args.seed_author:
        new_tax = seed_from_author(args.seed_author, tax,
                                   lookback_days=args.days or 365)
        save_taxonomy(new_tax)
        log.info("taxonomy.json updated from author papers.")
        return

    # -- resolve --topics: accepts top-level topic names OR sub-topic names --
    active_subtopics = None
    if args.topics:
        flat         = all_subtopics(tax)
        canon_sub    = {s.lower(): s for s in flat}                      # sub-topic lookup
        canon_top    = {t.lower(): t for t in tax}                       # top-level lookup
        active_subtopics = set()
        for req in args.topics:
            req_l = req.lower()
            if req_l in canon_top:
                # Expand top-level topic → all its sub-topics
                expanded = list(tax[canon_top[req_l]]["subtopics"].keys())
                active_subtopics.update(expanded)
                log.info("--topics %r expanded to: %s", req, expanded)
            elif req_l in canon_sub:
                active_subtopics.add(canon_sub[req_l])
            else:
                ap.error(
                    f"Unknown topic or sub-topic {req!r}. "
                    f"Run --list-topics to see all names."
                )
        log.info("Filtering to sub-topics: %s", active_subtopics)

    # -- calibrate --
    if args.calibrate:
        if not check_ollama():
            log.error("Ollama unreachable.")
            return
        subtopic_vecs, bg_vec = load_subtopic_embeddings(tax)
        calibrate_thresholds(tax, subtopic_vecs, bg_vec,
                             lookback_days=args.days or 7,
                             active_subtopics=active_subtopics)
        return

    REPORT_DIR.mkdir(exist_ok=True)
    topic_names = ", ".join(tax.keys())
    log.info("arXiv Digest — %s", topic_names)

    if args.reset_seen and SEEN_FILE.exists():
        SEEN_FILE.unlink()
        log.info("seen.json cleared")

    # Stage 1: fetch
    papers = fetch_feed(args.days, save_feed=args.save_feed)
    if not papers:
        render_report(datetime.now().strftime("%Y-%m-%d"), [], tax, all_papers=papers)
        log.info("nothing to do.")
        return

    # Stage 2: pre-filter — keyword match (broad net) then embedding triage
    # The keyword filter is intentionally loose here; it just culls obvious
    # non-EB/asteroseismology papers cheaply before we do any embedding work.
    seen      = load_seen()
    paper_db  = load_paper_db()
    subtopic_vecs, bg_vec = load_subtopic_embeddings(tax)
    log.info("STAGE 2 filter — %d seen | %d in paper DB | %d sub-topic vectors",
             len(seen), len(paper_db), len(subtopic_vecs))

    candidates   = []
    n_seen = n_nomatch = n_embed_reject = n_embed_accept = n_embed_review = 0
    for p in papers:
        if p["base_id"] in seen:
            n_seen += 1
            continue

        # Broad keyword pre-screen (fast, no LLM)
        hits = keyword_match(p, tax, active_subtopics)
        if not hits:
            n_nomatch += 1
            log.debug("  kw-drop  %s :: %s", p["base_id"], p["title"][:60])
            continue

        # Embedding triage — scores abstract against sub-topic vectors
        verdict, matched, scores = embed_triage(p, subtopic_vecs, bg_vec, active_subtopics)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        log.debug("  embed %s  top=%s  verdict=%s",
                  p["base_id"],
                  [(s[:25], f"{sc:.3f}") for s, sc in top],
                  verdict)

        if verdict == "reject":
            n_embed_reject += 1
            log.debug("  embed-drop %s :: %s", p["base_id"], p["title"][:60])
            continue

        # 'accept' skips Ollama triage; 'review' goes through it
        subtopics_hint = matched or hits
        candidates.append((p, subtopics_hint, verdict))
        if verdict == "accept":
            n_embed_accept += 1
            log.debug("  embed-ACCEPT %s :: %s", p["base_id"], p["title"][:60])
        else:
            n_embed_review += 1
            log.debug("  embed-REVIEW %s :: %s", p["base_id"], p["title"][:60])

    log.info("  %d candidate(s): %d embed-accept (no Ollama), %d embed-review, "
             "%d embed-drop, %d kw-drop, %d seen",
             len(candidates), n_embed_accept, n_embed_review,
             n_embed_reject, n_nomatch, n_seen)

    if not candidates:
        log.warning("No candidates — try --reset-seen or --days %d, or run --expand-taxonomy.",
                    args.days + 4)
        render_report(datetime.now().strftime("%Y-%m-%d"), [], tax)
        return

    if args.dry_run:
        log.info("STAGE 3-6 skipped (--dry-run). Candidates:")
        for p, subtopics_hint, verdict in candidates:
            log.info("  %-8s %s  %s  %s", verdict, p["base_id"],
                     subtopics_hint, p["title"][:60])
        return

    if not args.no_triage and not check_ollama():
        log.error("Ollama unreachable — aborting before triage.")
        return

    if args.limit:
        candidates = candidates[:args.limit]
        log.info("  limited to %d", len(candidates))

    # Stages 3-6: pipeline triage -> prefetch -> summarise
    #
    # Ollama is single-GPU so concurrent LLM calls don't help.
    # But fetch_fulltext is pure network I/O (~3-10s) and can run in a
    # background thread while Ollama summarises the previous paper (~10-25s).
    # _arxiv_get enforces a polite delay for HTML/PDF fulltext fetches across all threads via a lock.
    #
    # Timeline without pipeline:  [triage][fetch][summarise] [triage][fetch][summarise]
    # Timeline with pipeline:     [triage][fetch][summarise]
    #                                      [triage]     [fetch][summarise]
    # fetch is hidden inside the summarise window => ~30% faster on typical runs.

    flat       = all_subtopics(tax)
    kept       = []
    n_rejected = 0
    executor   = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fetch")

    # Pass 1: triage in batches — one Ollama call per TRIAGE_BATCH_SIZE papers
    # instead of one call per paper.  embed-accept papers skip triage entirely.
    pending = []  # list of (paper, subtopics, fetch_future)

    # Separate instant-accepts from papers needing Ollama
    need_triage  = [(p, hints) for p, hints, v in candidates if v != "accept"]
    instant_accept = [(p, hints) for p, hints, v in candidates if v == "accept"]

    # Kick off fetch immediately for embed-accepts while triage runs
    for p, hints in instant_accept:
        seen.add(p["base_id"])
        log.info("  embed-accept  %s", p["title"][:70])
        fut = executor.submit(fetch_fulltext, p)
        pending.append((p, hints, fut))

    if args.no_triage:
        for p, hints in need_triage:
            seen.add(p["base_id"])
            fut = executor.submit(fetch_fulltext, p)
            pending.append((p, hints[:1], fut))
    else:
        # Batch triage: chunk into groups of TRIAGE_BATCH_SIZE
        batches = [need_triage[i:i+TRIAGE_BATCH_SIZE]
                   for i in range(0, len(need_triage), TRIAGE_BATCH_SIZE)]
        log.info("  %d paper(s) → %d triage batch(es) of ≤%d",
                 len(need_triage), len(batches), TRIAGE_BATCH_SIZE)
        for b_idx, batch in enumerate(batches):
            log.info("  triage batch %d/%d (%d papers) …",
                     b_idx+1, len(batches), len(batch))
            results = batch_triage(batch, tax, paper_db)
            for (p, hints), (ok, subtopics) in zip(batch, results):
                seen.add(p["base_id"])
                if not ok:
                    n_rejected += 1
                    log.info("    rejected  %s", p["title"][:70])
                    continue
                log.info("    accepted  %s  %s", subtopics, p["title"][:60])
                fut = executor.submit(fetch_fulltext, p)
                pending.append((p, subtopics, fut))

    # Pass 2: summarise in order; each fut.result() blocks only if fetch
    # isn't done yet (usually it already is).
    for p, subtopics, fut in pending:
        parents = list({flat[s]["parent"] for s in subtopics if s in flat})
        parent  = parents[0] if parents else ""
        intro, methods, concl, source = fut.result()
        log.info("      summarise  source=%-12s  %s", source, subtopics)
        summary = summarize(p, intro, methods, concl, subtopics)
        add_to_paper_db(paper_db, p, subtopics, summary.get("summary", ""))
        kept.append({
            "paper": p, "summary": summary,
            "subtopics": subtopics, "parent_topic": parent,
            "source": source,
        })

    executor.shutdown(wait=True)

    save_seen(seen)
    save_paper_db(paper_db)
    log.info("Done — %d kept, %d rejected | paper DB now %d entries",
             len(kept), n_rejected, len(paper_db))

    date_str = datetime.now().strftime("%Y-%m-%d")
    out = render_report(date_str, kept, tax, all_papers=papers)
    log.info("Report: %s", out)
    try:
        webbrowser.open(out.as_uri())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TEMPLATES
# ---------------------------------------------------------------------------
REPORT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>arXiv Digest — {date}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#14161a; color:#e6e6e3;
    font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:860px; margin:0 auto; padding:32px 20px 80px; }}
  header {{ border-bottom:1px solid #2a2e35; padding-bottom:18px; margin-bottom:8px; }}
  h1 {{ font-size:24px; font-weight:600; margin:0 0 6px; }}
  .sub {{ color:#9aa0aa; font-size:14px; margin:0; }}
  .card {{ border:1px solid #2a2e35; border-radius:12px; padding:20px 24px;
    margin:24px 0; background:#191c21; }}
  .card h2 {{ font-size:18px; font-weight:600; margin:0 0 6px; line-height:1.4; }}
  .card h2 a {{ color:#e6e6e3; text-decoration:none; }}
  .card h2 a:hover {{ color:#7cc4ff; }}
  .authors {{ color:#9aa0aa; font-size:13px; margin:0 0 8px; }}
  .meta {{ font-size:12px; color:#7c828c; margin:0 0 14px; display:flex;
    flex-wrap:wrap; gap:6px; align-items:center; }}
  .topic   {{ background:#1a2e1a; color:#7fdb9e; padding:2px 9px; border-radius:10px; }}
  .subtopic{{ background:#1e2a3a; color:#85c8f0; padding:2px 9px; border-radius:10px; }}
  .src     {{ background:#222b3a; color:#85b7eb; padding:2px 9px; border-radius:10px; }}
  .summary {{ margin:0 0 10px; }}
  .methods {{ font-size:14px; color:#b8c8d8; margin:0 0 10px; }}
  .methods b {{ color:#c8d8e8; }}
  .findings {{ margin:0 0 12px; padding-left:20px; }}
  .findings li {{ margin:3px 0; }}
  .concl, .aux {{ font-size:14px; color:#c7ccd3; margin:6px 0; }}
  .concl b, .aux b {{ color:#e6e6e3; font-weight:600; }}
  .links {{ font-size:13px; margin:14px 0 0; }}
  .links a {{ color:#7cc4ff; text-decoration:none; margin-right:4px; }}
  .empty {{ color:#9aa0aa; text-align:center; padding:60px 0; }}
  
  .kw-appendix { border-top:2px solid #2a2e35; margin-top:48px; padding-top:24px; }
  .kw-appendix-title { font-size:18px; font-weight:600; display:flex;
    align-items:center; gap:12px; margin:0 0 20px; }
  .kw-toggle { background:#1e2a3a; color:#85c8f0; border:1px solid #2a4a6a;
    border-radius:8px; padding:3px 12px; cursor:pointer; font-size:13px; }
  .kw-toggle:hover { background:#2a3a4a; }
  .kw-count { color:#9aa0aa; font-size:14px; font-weight:400; }
  .kw-group { margin-bottom:28px; }
  .kw-heading { font-size:15px; font-weight:600; color:#7fdb9e;
    margin:0 0 10px; padding-bottom:6px; border-bottom:1px solid #2a2e35; }
  .kw-list { list-style:none; padding:0; margin:0; }
  .kw-item { padding:7px 4px; border-bottom:1px solid #1e2228;
    font-size:14px; display:flex; flex-wrap:wrap; gap:6px; align-items:baseline; }
  .kw-item a { color:#7cc4ff; text-decoration:none; flex:1 1 60%; }
  .kw-item a:hover { text-decoration:underline; }
  .kw-date { color:#7c828c; font-size:12px; white-space:nowrap; }
  .kw-sub { background:#1e2a3a; color:#85c8f0; padding:1px 7px;
    border-radius:8px; font-size:11px; }
  .kw-triggers { color:#6a7280; font-size:11px; font-style:italic; }

  a {{ color:#7cc4ff; }}
</style></head><body><div class="wrap">
<header>
  <h1>arXiv Digest — {date}</h1>
  <p class="sub">{count} paper(s) &middot; {topics} &middot; summarised locally by {model}</p>
</header>
{body}
</div></body></html>"""

INDEX_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>arXiv Digest — all reports</title>
<style>
  body {{ margin:0; background:#14161a; color:#e6e6e3;
    font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:620px; margin:0 auto; padding:40px 20px; }}
  h1 {{ font-size:22px; font-weight:600; }}
  ul {{ list-style:none; padding:0; }}
  li {{ border-bottom:1px solid #2a2e35; }}
  li a {{ display:block; padding:12px 4px; color:#7cc4ff; text-decoration:none; }}
  li a:hover {{ background:#191c21; }}
</style></head><body><div class="wrap">
<h1>arXiv Digest — all reports</h1>
<ul>{rows}</ul>
</div></body></html>"""


if __name__ == "__main__":
    main()