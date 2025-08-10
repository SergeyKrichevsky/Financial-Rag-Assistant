# Data_Processing_and_Indexing/chapters_enrich_v4.py
# Purpose: enrich existing chunked JSON with chapter metadata using the original .docx.
# It DOES NOT overwrite your input JSON; it writes a new JSON file.
# Usage (run from repo root):
#   python Data_Processing_and_Indexing/chapters_enrich_v4.py \
#       --json "Data_Processing_and_Indexing/book_metadata_merged_fixed.json" \
#       --docx "Data_Processing_and_Indexing/Final - The_6-Step_Personal_Finance_Reset_6x9_my_hand_breaks.docx" \
#       --out  "Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json"

import argparse, json, os, re, sys
from typing import List, Dict, Any, Tuple

# --- Helpers -----------------------------------------------------------------

def normalize_text(s: str) -> str:
    # Lowercase + collapse whitespace; keep only simple punctuation spacing comparable
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def looks_like_heading_text(txt: str) -> bool:
    t = txt.strip()
    if not t:
        return False
    # Heuristics: lines like "Chapter 1: ..." or "Step 3 – ..." etc.
    if re.match(r"^(chapter|step)\s*\d+([\s:\-\u2013].*)?$", t, flags=re.I):
        return True
    # Short & emphasized line can be a heading too (fallback)
    if len(t) <= 80 and t == t.upper() and re.search(r"[A-Z]", t):
        return True
    return False

def extract_headings_and_text(docx_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a normalized full-text of the docx and a list of headings with start offsets.
    Headings are primarily detected by paragraph style 'Heading*' (python-docx),
    with a text-pattern fallback if style is missing.
    """
    try:
        from docx import Document  # python-docx
    except Exception as e:
        print("[ERR] python-docx is required. Install with: pip install python-docx")
        raise

    doc = Document(docx_path)
    # We will build a normalized text and track offsets incrementally
    norm_parts: List[str] = []
    headings: List[Dict[str, Any]] = []

    offset = 0
    for para in doc.paragraphs:
        raw = para.text or ""
        norm = normalize_text(raw)
        # Style-based detection (preferred)
        style_name = ""
        try:
            style_name = getattr(para.style, "name", "") or ""
        except Exception:
            style_name = ""
        is_style_heading = style_name.startswith("Heading")  # e.g., "Heading 1", "Heading 2"

        # Fallback detection if styles are not reliable
        is_text_heading = looks_like_heading_text(raw)

        if is_style_heading or is_text_heading:
            # Record heading start at current offset BEFORE adding this paragraph text
            headings.append({
                "start": offset,
                "title": raw.strip(),
                "style": style_name or ("fallback" if is_text_heading else "")
            })

        # Append text
        norm_parts.append(norm)
        # +1 for newline separator we add between paragraphs
        offset += len(norm) + 1

    full_norm_text = "\n".join(norm_parts)

    # Assign chapter numbers sequentially
    for i, h in enumerate(headings, 1):
        h["chapter_number"] = i

    return full_norm_text, headings

def find_chunk_pos_in_text(chunk_text: str, doc_text_norm: str) -> int:
    """
    Try to locate a chunk inside normalized document text using decreasing snippet sizes.
    Returns character offset in normalized doc_text or -1 if not found.
    """
    base = normalize_text(chunk_text)
    # Progressive attempts: try larger -> smaller snippets
    for max_chars in (220, 160, 120, 90, 60, 40):
        snippet = base[:max_chars].strip()
        if len(snippet) < 15:  # too short to be reliable
            continue
        idx = doc_text_norm.find(snippet)
        if idx != -1:
            return idx
    # Try first N words as another heuristic
    words = base.split()
    if len(words) >= 8:
        snippet = " ".join(words[:12])
        idx = doc_text_norm.find(snippet)
        if idx != -1:
            return idx
    return -1

def assign_chapter(pos: int, headings: List[Dict[str, Any]]) -> Tuple[str, int]:
    """
    Given a position in doc_text, return (chapter_title, chapter_number)
    corresponding to the nearest heading whose start <= pos. If none, return ("Unknown", 0).
    """
    if pos < 0 or not headings:
        return "Unknown", 0
    # Binary search-like linear pass (headings count should be small)
    last_title, last_num = "Unknown", 0
    for h in headings:
        if h["start"] <= pos:
            last_title, last_num = h["title"], h["chapter_number"]
        else:
            break
    return last_title, last_num

# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to existing chunked JSON")
    ap.add_argument("--docx", required=True, help="Path to original .docx")
    ap.add_argument("--out",  required=True, help="Path to write enriched JSON (new file)")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        print(f"[ERR] JSON not found: {args.json}"); sys.exit(1)
    if not os.path.exists(args.docx):
        print(f"[ERR] DOCX not found: {args.docx}"); sys.exit(1)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load JSON records
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        print("[ERR] Input JSON should be a non-empty array of objects."); sys.exit(1)

    # Extract normalized doc text + headings with offsets
    print("[..] Reading DOCX and extracting headings (style 'Heading*' with fallback heuristics)...")
    doc_text_norm, headings = extract_headings_and_text(args.docx)
    print(f"[OK] Headings detected: {len(headings)}")
    if headings:
        print("     First headings:", [h.get("title") for h in headings[:3]])

    enriched = []
    last_known_title, last_known_num = "Unknown", 0

    for i, obj in enumerate(data):
        text = obj.get("text", "")
        mode = obj.get("mode") or obj.get("category")  # we will map to 'category'
        pos = find_chunk_pos_in_text(text, doc_text_norm)

        if pos != -1:
            chapter_title, chapter_number = assign_chapter(pos, headings)
            last_known_title, last_known_num = chapter_title, chapter_number
        else:
            # Fallback: forward-fill last known chapter in reading order
            chapter_title, chapter_number = last_known_title, last_known_num

        new_obj = dict(obj)  # keep original fields
        # Enriched metadata:
        new_obj["chapter_title"]  = chapter_title
        new_obj["chapter_number"] = chapter_number
        new_obj["category"]       = mode if mode is not None else obj.get("category", None)
        new_obj["position"]       = i
        new_obj["source_id"]      = "finance_book_v4"

        enriched.append(new_obj)

    # Save to a NEW file
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    # Quick summary
    unknown = sum(1 for r in enriched if (r.get("chapter_number", 0) == 0 or r.get("chapter_title") == "Unknown"))
    print(f"[DONE] Wrote: {args.out}")
    print(f"       Total chunks: {len(enriched)} | Unknown chapter: {unknown}")

if __name__ == "__main__":
    main()
