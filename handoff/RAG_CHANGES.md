# Changes to `rag_helper.py`

## The problem

`get_baseline_knowledge()` loads files top-to-bottom until it hits `max_chars` (30,000), then stops mid-list. The knowledge base is 51,783 chars across five files, so with the current cap **files 03 and 04 would never load** — and they contain the formula library and the modes.

There is also a silent-truncation bug: the inner loop uses `break` when a chunk would exceed the cap, which exits that file's loop but *continues to the next file*. So you get partial file A, partial file B, etc., with no warning. Nothing tells you what got dropped.

## Change 1 — replace the `universal_essentials` list

```python
    universal_essentials = [
        # Superlunar knowledge base — load in this exact order.
        "00_operating_instructions.txt",   # calibration, two layers, payoff-last, honesty
        "01_spine_and_layers.txt",         # hook/promise/tension/delivery, structure, execution, goals
        "02_mechanisms.txt",               # promise + tension taxonomies, glossary
        "03_formula_library.txt",          # six formats + reusable template
        "04_modes_and_metrics.txt",        # five modes, metrics, workflow, worked examples

        # Supporting material
        "video_retention.txt",
        "architecture_of_retention.txt",
        "hook_mechanisms.txt",
        "failuretofix.txt",
        "50_Hook_Examples.pdf",
        "HookWritingGuide_Download.pdf",
        "Trial Reels Guide.pdf",
        "x8u4vlfmj1n62gdem7rbpyq52jcg.pdf",
    ]
```

**Removed:** `master.txt` (superseded — delete the file from `knowledge/` so it can't be re-ingested) and `thisvsthat.txt` (see note at bottom).

## Change 2 — raise the baseline cap

The five core files are 51,783 chars. In `retrieve_smart_context`:

```python
    # Step 1: ALWAYS get the full core knowledge base (55K chars)
    baseline_text, baseline_cits = get_baseline_knowledge(meta, 55000)

    # Step 2: Add goal-specific knowledge (35K chars)
    specific_text, specific_cits = get_specific_knowledge(
        meta, mat, transcript, frames, creator_note, goal, 35000
    )
```

And raise the signature default to match: `def retrieve_smart_context(..., max_chars: int = 95000)`.

Note the final `return combined[:max_chars], all_citations` — that hard-truncates the *combined* string, so `max_chars` must be ≥ baseline + specific or the tail gets silently cut.

If 95K is too much context for your latency or cost budget, cut the *supporting* files from the essentials list rather than lowering the cap. The five core files should never be partially loaded.

## Change 3 — make truncation loud instead of silent

Replace the inner loop in `get_baseline_knowledge`:

```python
    CORE_FILES = {
        "00_operating_instructions.txt",
        "01_spine_and_layers.txt",
        "02_mechanisms.txt",
        "03_formula_library.txt",
        "04_modes_and_metrics.txt",
    }
    truncated = []
    missing = []

    for target_file in universal_essentials:
        file_chunks = []
        for idx, m in meta.items():
            if m["file"] == target_file:
                file_chunks.append((m["chunk_id"], m["text"]))

        if not file_chunks:
            missing.append(target_file)
            continue

        file_chunks.sort(key=lambda x: x[0])

        for chunk_id, text in file_chunks:
            if total_chars + len(text) > max_chars:
                truncated.append(target_file)
                break
            baseline_chunks.append(f"[Essential: {target_file}]\n{text}")
            baseline_citations.append({
                "file": target_file,
                "type": "universal_essential",
                "chunk_id": chunk_id
            })
            total_chars += len(text)

    if missing:
        print(f"[BASELINE][MISSING] Not in index (run ingest_knowledge.py): {', '.join(missing)}")
    if truncated:
        print(f"[BASELINE][TRUNCATED] Hit {max_chars} cap, cut short: {', '.join(set(truncated))}")
        core_cut = CORE_FILES & set(truncated)
        if core_cut:
            print(f"[BASELINE][CRITICAL] Core knowledge truncated: {', '.join(core_cut)} — raise max_chars.")
```

## Change 4 — re-ingest

```bash
rm knowledge/master.txt knowledge/thisvsthat.txt
cp 00_*.txt 01_*.txt 02_*.txt 03_*.txt 04_*.txt knowledge/
python ingest_knowledge.py
```

Then run one analysis and check the logs for `[BASELINE][CRITICAL]` or `[BASELINE][MISSING]`.

## Note on `thisvsthat.txt`

Its genuinely useful content — the VS sub-type→goal map, the three reveal structures, the real failure modes — is already absorbed into §5.1 of `03_formula_library.txt`. What's left is fabricated performance ranges ("Humor VS 500K–5M"), virality guarantees ("virtually guarantees above-average performance"), and the claim that structure predicts view count. Loading it as a *universal essential* means those claims arrive on every analysis with the same authority as your real doctrine, and they directly contradict the calibration rules in `00_operating_instructions.txt`. Recommend removing it. If you want to keep it for reference, move it out of the essentials list so it's only reachable by semantic search.

## Optional follow-up: chunk sizing

Because `get_baseline_knowledge` is file-based rather than semantic, chunk boundaries matter less than they would in a normal RAG — whole files load in order. But if `ingest_knowledge.py` chunks small (<1000 chars), the per-chunk `[Essential: filename]` prefix repeats a lot and eats budget. Worth a look if context is tight.
