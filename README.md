<div align="center">

<img src="logo.png" alt="arXiv Digest logo" width="130"/>

# arXiv Digest

**Your own daily arXiv reader — fully local.**

It grabs the newest `astro-ph.SR` papers, keeps the ones on your topics, summarises
them with a local model, and writes a tidy HTML report that opens in your browser.
No API keys, no cloud, nothing leaves your machine.

</div>

---

## What it does

Run it once a day and you get a dated report of the papers worth your attention,
each with a short summary, the methods, the key numbers, and why it matters. Papers
you've already seen are skipped, so every run only deals with what's new.

It's built around a small idea: don't make the language model read everything.
A cheap filter throws out the obvious misses, embeddings catch the clear hits, and
the model only weighs in on the borderline cases. Same for the text — it reads the
abstract, intro, methods, and conclusions, not the whole PDF. So a daily run is
quick even on a laptop.

Paper metadata is fetched via arXiv's **OAI-PMH interface**
(`https://oaipmh.arxiv.org/oai`) rather than the rate-limited REST/Atom API.
OAI-PMH is the official bulk-harvesting endpoint — no rate limits, resumption-token
paging, and precise `from`-date filtering. The only outbound traffic is arXiv and
your local Ollama server.

```
arXiv (OAI-PMH)  →  keyword filter  →  embedding triage  →  (borderline only) LLM check
                 →  fetch text  →  LLM summary  →  HTML report
```

## Quick start

You'll need Python 3.9+ and [Ollama](https://ollama.com/download) with two models:

```bash
ollama pull qwen2.5:14b        # writes the summaries
ollama pull nomic-embed-text   # does the fast relevance triage
```

Then:

```bash
chmod +x setup.sh
./setup.sh                     # installs deps, finds your model, warms it up
python3 arxiv_digest.py
```

First run writes a starter topic list, builds its embeddings, and opens the report.
Reports pile up in `./reports/`, with `index.html` listing them all.

> **Dependencies:** only `requests` and `beautifulsoup4` are required.
> OAI-PMH XML parsing uses Python's built-in `xml.etree.ElementTree` — no
> `feedparser` needed.

## Picking your topics

Topics live in `taxonomy.json` — a simple tree of subjects, each with a description
and some keywords. It ships covering eclipsing binaries and asteroseismology, but
it's just a file: edit it to whatever you follow.

The list isn't fixed. As you run the tool it keeps a record of what it's classified
(`papers.json`) and uses that to stay consistent. And you can let the model suggest
new topics it notices cropping up:

```bash
python3 arxiv_digest.py --expand-taxonomy
```

## Everyday use

```text
python3 arxiv_digest.py                  the daily run
python3 arxiv_digest.py --days 7         look further back
python3 arxiv_digest.py --topics "Pulsating stars in EBs"   just one subject
python3 arxiv_digest.py --limit 5        only summarise the first few
python3 arxiv_digest.py --dry-run        see what would pass, skip the LLM
python3 arxiv_digest.py --list-topics    print the topic tree
python3 arxiv_digest.py -v               verbose logging
```

Every stage logs how many papers went in and came out, so if a run turns up empty
you can see exactly where the feed dried up.

## If the filter feels off

The relevance check compares each paper to your topics *against* a generic-astro
baseline, so papers that just mention the right words in passing don't sneak through.
If too much or too little is getting in, let it suggest better thresholds:

```bash
python3 arxiv_digest.py --calibrate --days 7
```

It prints where recent papers actually scored and the numbers to drop into the
config block at the top of `arxiv_digest.py` (which is also where you set the model,
the categories, and the lookback window).

## Good to know

- Papers are fetched via OAI-PMH, arXiv's official bulk-harvesting interface.
  There are no per-request rate limits; the tool uses a short polite delay between
  resumption-token pages only.
- `--save-feed` dumps each OAI-PMH page as `feed_debug_p0.xml`, `feed_debug_p1.xml`, …
- PDF reading is optional — uncomment `PyMuPDF` in `requirements.txt` to enable the
  fallback for papers without an HTML version.
- Everything is local; the only external host it contacts is `oaipmh.arxiv.org`
  (metadata) and `arxiv.org` (HTML/PDF fulltext).

## License

MIT