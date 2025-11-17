import os
import re
import sys
import math
import pathlib
import itertools
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import typer
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from rich.prompt import Confirm
from dotenv import load_dotenv
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

import chromadb
from chromadb import Client
from chromadb.config import Settings


# Lazy imports for optional providers
def _lazy_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for provider=sentence-transformers. Install it first."
        ) from exc


try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
load_dotenv()

# Global speaker label pattern (matches diarized labels like speaker_0: and names like Robert Reese:)
# Allow zero or more whitespace after the colon to be robust across files
SPEAKER_RE = re.compile(r"^\s*(?:speaker_\d+|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*):\s*")

# Cache for per-record speaker attribution maps
SPEAKER_MAP_CACHE: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}


# -----------------------------
# Utilities
# -----------------------------


def to_absolute_path(path_str: str) -> str:
    return str(pathlib.Path(path_str).expanduser().resolve())


def default_data_root() -> str:
    return to_absolute_path("../Data/Transcripts")


def default_persist_dir() -> str:
    return "ceo_forum_cloud"  # Cloud database identifier


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_filename_title(name: str) -> str:
    base = pathlib.Path(name).stem
    base = re.sub(r"\s*\(\d+p\)$", "", base)  # Drop resolutions like (720p)
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base).strip()
    return base[:200]


def strip_line_number_prefixes(text: str) -> str:
    # Some previews show L123: prefixes; remove them if present
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        cleaned.append(re.sub(r"^L\d+:", "", line))
    return "\n".join(cleaned)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    # collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_leading_direct_answer(text: str) -> str:
    """Remove a leading 'Direct answer' label (and common Markdown variants).

    Handles forms like:
    - Direct answer\n...
    - Direct answer: ...
    - **Direct answer**: ...
    - ### Direct answer\n...
    - Direct Answer — ...
    Only affects the very beginning of the text.
    """
    if not text:
        return text
    # Work on a left-trimmed view to match headings at the start
    view = text.lstrip()
    # Case 1: Heading on its own line, then newline(s)
    pattern_line = r"^(?:#{1,6}\s*)?(?:\*\*|__)?\s*direct\s+answer\s*(?:\*\*|__)?\s*[:\-–—]?\s*\n+"
    new_view = re.sub(pattern_line, "", view, count=1, flags=re.IGNORECASE)
    if new_view != view:
        return new_view.lstrip()
    # Case 2: Inline prefix with punctuation then space (e.g., 'Direct answer: ')
    pattern_inline = r"^(?:#{1,6}\s*)?(?:\*\*|__)?\s*direct\s+answer\s*(?:\*\*|__)?\s*[:\-–—]\s+"
    new_view2 = re.sub(pattern_inline, "", view, count=1, flags=re.IGNORECASE)
    if new_view2 != view:
        return new_view2.lstrip()
    return text


def preprocess_turns(text: str) -> str:
    """Ensure double-newline separated speaker turns while preserving speaker labels.

    - Drops any header before the first speaker line
    - Collapses excess blank lines
    - Keeps lines beginning with a speaker label as separate turns
    """
    # Robustly drop any header before the first speaker label
    m = re.search(r"(?m)^\s*(?:speaker_\d+|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*):\s*", text)
    if m:
        text = text[m.start():]

    lines = [ln.rstrip() for ln in text.splitlines()]

    turns: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if SPEAKER_RE.match(ln):
            if buf:
                turns.append(" ".join(part for part in buf if part))
                buf = []
            buf.append(ln.strip())
        else:
            # continuation of prior turn or noise; append to buffer
            if ln.strip():
                buf.append(ln.strip())
    if buf:
        turns.append(" ".join(part for part in buf if part))

    # Join turns with double newline
    normalized = "\n\n".join(turns)
    # Final whitespace normalization
    normalized = normalize_whitespace(normalized)
    return normalized


def split_into_turns(text: str) -> List[str]:
    """Return a list of speaker turns (after preprocess_turns)."""
    prepped = preprocess_turns(text)
    # Turns are separated by double newlines after preprocessing
    return [t for t in prepped.split("\n\n") if t.strip()]


def extract_speaker_label(turn_text: str) -> Optional[str]:
    m = re.match(r"^\s*(speaker_\d+|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*):\s*", turn_text)
    if m:
        return m.group(1)
    return None


def expand_chunk_to_full_turn(
    chunk_text: str,
    source_path: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Given a chunk snippet and its source file, return the full speaker turn that best contains it,
    along with immediate neighbor turns (previous and next) for context expansion.
    """
    try:
        raw = read_text_file(source_path)
        raw = strip_line_number_prefixes(raw)
        text = normalize_whitespace(raw)
    except Exception:
        # If anything fails, fall back to the original chunk
        return chunk_text, extract_speaker_label(chunk_text), None, None

    turns = split_into_turns(text)
    if not turns:
        return chunk_text, extract_speaker_label(chunk_text), None, None

    # Use partial_ratio to match even if the chunk is truncated
    best = rf_process.extractOne(
        chunk_text,
        turns,
        scorer=rf_fuzz.partial_ratio,
    )
    if not best:
        return chunk_text, extract_speaker_label(chunk_text), None, None

    best_turn, score, best_idx = best
    # Require a minimal match quality; otherwise keep original
    if score < 50:
        return chunk_text, extract_speaker_label(chunk_text), None, None

    prev_turn = turns[best_idx - 1].strip() if isinstance(best_idx, int) and best_idx is not None and best_idx > 0 else None
    next_turn = turns[best_idx + 1].strip() if isinstance(best_idx, int) and best_idx is not None and best_idx + 1 < len(turns) else None

    return best_turn.strip(), extract_speaker_label(best_turn), prev_turn, next_turn


def _is_filler_turn(text: str) -> bool:
    t = (text or "").strip().lower()
    # Very short acknowledgements or host lead-ins
    if len(t) <= 80:
        fillers = [
            "right.",
            "right!",
            "right",
            "yeah",
            "yeah.",
            "ok",
            "ok.",
            "okay",
            "okay.",
            "uh",
            "uh-huh",
            "mm-hmm",
            "great",
            "great.",
            "interesting.",
            "absolutely.",
            "definitely.",
        ]
        if t in fillers:
            return True
    # Generic short setup statements
    patterns = [
        r"^that's (great|exciting|interesting)",
        r"^good question",
        r"^big challenge",
        r"^let'?s talk about",
    ]
    for pat in patterns:
        if re.search(pat, t):
            return True
    # Extremely short length with no punctuation suggesting low substance
    if len(t) < 120 and not re.search(r"[\.!?]", t):
        return True
    return False


def _is_question(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip()
    if t.endswith("?"):
        return True
    tl = t.lower()
    return bool(re.search(r"\b(how|what|why|when|where|which|who|whom|whose|can you|could you|would you)\b", tl))


def _detect_host_labels(speaker_map: Dict[str, Dict[str, Optional[str]]]) -> List[str]:
    hosts: List[str] = []
    for lbl, info in (speaker_map or {}).items():
        role = (info.get("role") or "").lower()
        display = (info.get("display") or "").lower()
        if "host" in role or "interview" in role or "host" in display or "interview" in display:
            hosts.append(lbl)
    return hosts


def normalize_speaker_key(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    key = label.strip().rstrip(":")
    key = key.replace("SPEAKER_", "speaker_")
    key = key.lower()
    return key


def parse_json_safely(text: str) -> Optional[Dict]:
    import json as _json

    # Try whole text
    try:
        return _json.loads(text)
    except Exception:
        pass
    # Try to find a JSON block between braces
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return _json.loads(m.group(0))
        except Exception:
            pass
    # Try fenced code blocks
    m = re.search(r"```(?:json)?\n([\s\S]*?)\n```", text)
    if m:
        try:
            return _json.loads(m.group(1))
        except Exception:
            pass
    return None


def infer_speaker_map_oai(
    record_id: str,
    source_path: str,
    meta_hint: Dict[str, Optional[str]],
    model: str = "gpt-5-mini",
) -> Dict[str, Dict[str, Optional[str]]]:
    """Infer mapping from diarized labels (speaker_0, SPEAKER_1, etc.) to real names/roles.

    Returns dict like:
      {
        "speaker_0": {"name": "Robert Reiss", "role": "Host", "org": "The CEO Forum", "display": "Robert Reiss (Host, The CEO Forum)"},
        "speaker_1": {"name": "Mike Critelli", "role": "Former CEO", "org": "Pitney Bowes", "display": "Mike Critelli, Former CEO of Pitney Bowes"}
      }
    """
    # Cache check
    if record_id in SPEAKER_MAP_CACHE:
        return SPEAKER_MAP_CACHE[record_id]

    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        SPEAKER_MAP_CACHE[record_id] = {}
        return {}

    # Read and trim transcript
    try:
        raw = read_text_file(source_path)
        raw = strip_line_number_prefixes(raw)
        prepped = preprocess_turns(normalize_whitespace(raw))
    except Exception:
        SPEAKER_MAP_CACHE[record_id] = {}
        return {}

    excerpt_head = prepped[:12000]
    excerpt_tail = prepped[-4000:]

    guest_hint = (meta_hint.get("guest") or "").strip()
    title_hint = (meta_hint.get("title") or "").strip()
    file_hint = os.path.basename(source_path)
    category_hint = meta_hint.get("category") or ""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_msg = (
        "You are an expert at identifying speakers in CEO interview transcripts. "
        "Map diarized speaker labels (e.g., speaker_0, SPEAKER_1) to real names and full executive titles. "
        "Focus on identifying the CEO/executive guest and the host/interviewer. "
        "Extract complete titles (e.g., 'CEO', 'Former CEO', 'Chairman & CEO', 'Founder') and full company names. "
        "Output pure JSON mapping from label to {name, role, org, display}."
    )
    user_msg = (
        "Analyze this CEO interview transcript to identify all speakers.\n\n"
        "Context clues:\n"
        f"- Guest: {guest_hint or 'unknown'}\n"
        f"- Title: {title_hint or 'unknown'}\n"
        f"- File: {file_hint}\n"
        f"- Category: {category_hint or 'unknown'}\n\n"
        "Look for:\n"
        "1. Self-introductions (\"I'm [Name], [Title] at [Company]\")\n"
        "2. The host welcoming the guest (\"Today we're joined by [Name]...\")\n"
        "3. Discussion of the guest's company or role\n"
        "4. Signature phrases from known hosts (Robert Reiss, etc.)\n\n"
        "Return JSON in this exact format:\n"
        "{ '<label>': { 'name': str, 'role': str|null, 'org': str|null, 'display': str } }\n\n"
        "Example output:\n"
        "{\n"
        "  \"speaker_0\": {\"name\": \"Robert Reiss\", \"role\": \"Host\", \"org\": \"The CEO Forum\", \"display\": \"Robert Reiss (Host, The CEO Forum)\"},\n"
        "  \"speaker_1\": {\"name\": \"Mary Barra\", \"role\": \"CEO\", \"org\": \"General Motors\", \"display\": \"Mary Barra, CEO of General Motors\"}\n"
        "}\n\n"
        "--- TRANSCRIPT BEGINNING ---\n" + excerpt_head + "\n--- END BEGINNING ---\n\n"
        "--- TRANSCRIPT ENDING ---\n" + excerpt_tail + "\n--- END ENDING ---\n\n"
        "Return ONLY the JSON object, no other text."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=400,
        )
        text = resp.choices[0].message.content or "{}"
        data = parse_json_safely(text) or {}
        # Normalize keys
        norm: Dict[str, Dict[str, Optional[str]]] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                nk = normalize_speaker_key(str(k))
                if not nk:
                    continue
                if not isinstance(v, dict):
                    continue
                name = v.get("name") if isinstance(v.get("name"), str) else None
                role = v.get("role") if isinstance(v.get("role"), str) else None
                org = v.get("org") if isinstance(v.get("org"), str) else None
                display = v.get("display") if isinstance(v.get("display"), str) else None
                # Build a reasonable display if missing
                if not display and name:
                    if role and org:
                        display = f"{name}, {role} at {org}"
                    elif org:
                        display = f"{name}, {org}"
                    elif role:
                        display = f"{name}, {role}"
                    else:
                        display = name
                norm[nk] = {"name": name, "role": role, "org": org, "display": display}
        SPEAKER_MAP_CACHE[record_id] = norm
        return norm
    except Exception:
        SPEAKER_MAP_CACHE[record_id] = {}
        return {}


def _recursive_split(
    text: str,
    separators: List[str],
    max_chars: int,
    overlap_chars: int,
) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    if not separators:
        # hard split with overlap
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap_chars
            if start < 0:
                start = 0
        return chunks

    sep = separators[0]
    parts = text.split(sep)
    if len(parts) == 1:
        # cannot split by this sep, try next
        return _recursive_split(text, separators[1:], max_chars, overlap_chars)

    # Greedily rebuild chunks from parts, then recursively fix overflows
    rebuilt: List[str] = []
    current = parts[0]
    for part in parts[1:]:
        candidate = current + sep + part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            rebuilt.append(current)
            # start new chunk with overlap from end of previous
            tail = current[-overlap_chars:]
            current = tail + sep + part
    if current:
        rebuilt.append(current)

    # Ensure each chunk under limit by deeper splitting if needed
    final_chunks: List[str] = []
    for ch in rebuilt:
        if len(ch) > max_chars:
            final_chunks.extend(_recursive_split(ch, separators[1:], max_chars, overlap_chars))
        else:
            final_chunks.append(ch)

    # Apply explicit overlap between consecutive chunks
    with_overlap: List[str] = []
    if final_chunks:
        with_overlap.append(final_chunks[0])
        for i in range(1, len(final_chunks)):
            prev = final_chunks[i - 1]
            cur = final_chunks[i]
            prefix = prev[-overlap_chars:]
            if not cur.startswith(prefix):
                cur = prefix + cur
            with_overlap.append(cur)
    return [c.strip() for c in with_overlap if c.strip()]


def chunk_text_turn_aware(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 300,
) -> List[str]:
    # 1) Preprocess to double-newline-separated speaker turns
    prepped = preprocess_turns(text)
    # 2) Recursive splitting with priority separators
    separators = ["\n\n", "\n", ". ", " "]
    chunks = _recursive_split(prepped, separators, max_chars, overlap_chars)
    return chunks


@app.command()
def preview(
    file_path: str = typer.Argument(..., help="Path to a single transcript .txt file"),
    max_chars: int = typer.Option(2000, help="Max characters per chunk."),
    overlap_chars: int = typer.Option(200, help="Character overlap between chunks."),
    json_output: bool = typer.Option(True, help="Print structured JSON output."),
):
    """Preview turn-aware chunks for a single transcript without indexing."""
    file_path_abs = to_absolute_path(file_path)
    raw = read_text_file(file_path_abs)
    raw = strip_line_number_prefixes(raw)
    text = normalize_whitespace(raw)
    prepped = preprocess_turns(text)

    chunks = chunk_text_turn_aware(prepped, max_chars=max_chars, overlap_chars=overlap_chars)

    # derive simple metadata
    md = extract_basic_metadata(prepped, file_path_abs)
    filename = os.path.basename(file_path_abs)
    category = md.get("category")
    source = filename

    # speaker label detector
    def begins_with_speaker_label(s: str) -> bool:
        first_line = s.splitlines()[0] if s else ""
        return bool(SPEAKER_RE.match(first_line))

    out = {
        "file": file_path_abs,
        "filename": filename,
        "category": category,
        "source": source,
        "source_type": category,
        "chunks": [
            {
                "chunk_index": i,
                "num_chars": len(c),
                "begins_with_speaker_label": begins_with_speaker_label(c),
                "preview": c
            }
            for i, c in enumerate(chunks)
        ],
        "total_chunks": len(chunks),
    }

    if json_output:
        sys.stdout.write(orjson.dumps(out, option=orjson.OPT_INDENT_2).decode("utf-8") + "\n")
    else:
        table = Table(title=f"Preview: {source}")
        table.add_column("#", justify="right")
        table.add_column("Chars", justify="right")
        table.add_column("Speaker Start")
        table.add_column("Preview")
        for ch in out["chunks"]:
            table.add_row(str(ch["chunk_index"]), str(ch["num_chars"]), "Yes" if ch["begins_with_speaker_label"] else "No", ch["preview"])
        console.print(table)


def extract_basic_metadata(text: str, file_path: str) -> Dict[str, Optional[str]]:
    title = clean_filename_title(os.path.basename(file_path))
    category = pathlib.Path(file_path).parts
    # Find the immediate category under Data/Transcripts/{category}
    cat = None
    try:
        idx = category.index("Transcripts")
        if idx + 1 < len(category):
            cat = category[idx + 1]
    except ValueError:
        cat = None

    # Heuristic extraction of possible names (very light)
    header = text[:2000]
    possible_names: List[str] = []
    patterns = [
        r"we(?:'| a)re here (?:today )?with ([A-Z][a-z]+\s+[A-Z][a-zA-Z\-']+)",
        r"with ([A-Z][a-z]+\s+[A-Z][a-zA-Z\-']+)",
        r"I(?:'| a)m ([A-Z][a-z]+\s+[A-Z][a-zA-Z\-']+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, header):
            name = m.group(1).strip()
            if name not in possible_names:
                possible_names.append(name)

    guest = possible_names[0] if possible_names else None

    return {
        "title": title,
        "category": cat,
        "guest": guest,
    }


def walk_transcript_txts(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                files.append(to_absolute_path(os.path.join(dirpath, fn)))
    files.sort()
    return files


# -----------------------------
# Embeddings Providers
# -----------------------------


class EmbeddingProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        SentenceTransformer = _lazy_import_sentence_transformers()
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # sentence-transformers returns numpy arrays; convert to lists for Chroma
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-large") -> None:
        if OpenAI is None:
            raise RuntimeError("openai python package not installed; cannot use provider=openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=api_key)
        # Map common aliases to canonical names
        alias = (model_name or "").strip()
        alias = alias.replace(":", "-")
        if alias in {"embeddings-3-small", "embedding-3-small", "text-embedding-3-small"}:
            self.model_name = "text-embedding-3-small"
        elif alias in {"embeddings-3-large", "embedding-3-large", "text-embedding-3-large"}:
            self.model_name = "text-embedding-3-large"
        else:
            self.model_name = model_name

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        # Batch to stay under token limits; 2048 chars per input is safe for most batches
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            for d in resp.data:
                embeddings.append(d.embedding)
        return embeddings


def build_embedding_provider(provider: str, model: Optional[str]) -> EmbeddingProvider:
    provider = provider.lower()
    if provider in {"sentence-transformers", "sbert", "hf"}:
        return SentenceTransformersEmbeddingProvider(model_name=model or "all-MiniLM-L6-v2")
    if provider in {"openai", "oai"}:
        return OpenAIEmbeddingProvider(model_name=model or "text-embedding-3-large")
    raise ValueError(f"Unsupported embeddings provider: {provider}")


# -----------------------------
# Chroma Helpers
# -----------------------------


def get_chroma_client(persist_directory: str) -> Client:
    # Use Chroma Cloud instead of local persistent client
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")
    
    if not api_key or not tenant:
        raise RuntimeError("CHROMA_API_KEY and CHROMA_TENANT must be set in environment variables")
    
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )
    return client


def get_or_create_collection(client: Client, name: str) -> chromadb.api.models.Collection.Collection:
    try:
        coll = client.get_collection(name)
    except Exception:
        coll = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return coll


# -----------------------------
# Index Command
# -----------------------------


def load_metadata_manifest(manifest_path: str = "../Data/metadata_manifest.json") -> Dict:
    """Load the enriched metadata manifest."""
    abs_path = to_absolute_path(manifest_path)
    if not os.path.exists(abs_path):
        console.log(f"[yellow]Warning: Metadata manifest not found at {abs_path}[/yellow]")
        console.log("[yellow]Run 'python enrich_metadata.py' first for better metadata[/yellow]")
        return {}
    
    try:
        import json as _json
        with open(abs_path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception as e:
        console.log(f"[yellow]Warning: Could not load metadata manifest: {e}[/yellow]")
        return {}


@app.command()
def index(
    data_root: str = typer.Option(default_data_root(), help="Root folder containing transcript categories."),
    persist_dir: str = typer.Option(default_persist_dir(), help="Chroma persistence directory."),
    collection: str = typer.Option("transcripts", help="Chroma collection name."),
    max_chars: int = typer.Option(2000, help="Max characters per chunk."),
    overlap_chars: int = typer.Option(200, help="Character overlap between chunks (~10-20%)."),
    reindex: bool = typer.Option(False, help="If true, clears existing collection before reindexing."),
    skip_unchanged: bool = typer.Option(True, help="Skip files whose content hash hasn't changed."),
    files: List[str] = typer.Option(None, help="Optional list of specific .txt files to index (absolute or relative paths)."),
    manifest_path: str = typer.Option("../Data/metadata_manifest.json", help="Path to enriched metadata manifest."),
):
    """Build the vector index from transcript .txt files."""
    data_root = to_absolute_path(data_root)
    persist_dir = to_absolute_path(persist_dir)
    console.log(f"Indexing from {data_root} → {persist_dir} (collection={collection})")

    # Load enriched metadata manifest
    enriched_manifest = load_metadata_manifest(manifest_path)
    if enriched_manifest:
        console.log(f"Loaded enriched metadata for {len(enriched_manifest)} files")
    
    # Always use OpenAI text-embedding-3-small for indexing
    provider_obj = OpenAIEmbeddingProvider(model_name="text-embedding-3-small")
    client = get_chroma_client(persist_dir)

    if reindex:
        try:
            client.delete_collection(collection)
            console.log(f"Deleted existing collection: {collection}")
        except Exception:
            pass

    coll = get_or_create_collection(client, collection)

    file_list: List[str]
    if files:
        # Normalize paths and filter to .txt that exist
        normalized: List[str] = []
        for p in files:
            ap = to_absolute_path(p)
            if os.path.isfile(ap) and ap.lower().endswith(".txt"):
                normalized.append(ap)
        file_list = sorted(set(normalized))
    else:
        file_list = walk_transcript_txts(data_root)

    if not file_list:
        rprint("[bold red]No .txt transcripts found.[/bold red]")
        raise typer.Exit(code=1)

    console.log(f"Found {len(file_list)} transcript files")

    # Process each file incrementally, skipping unchanged if configured
    total_chunks = 0
    for file_idx, file_path in enumerate(file_list):
        raw = read_text_file(file_path)
        raw = strip_line_number_prefixes(raw)
        text = normalize_whitespace(raw)
        if not text:
            continue

        # Turn-aware preprocessing yields stable content hash per file
        prepped = preprocess_turns(text)
        file_hash = hashlib_safe(prepped)

        md = extract_basic_metadata(prepped, file_path)
        record_id = f"rec::{hashlib_safe(file_path)}"
        title = md.get("title") or os.path.basename(file_path)
        
        # Look up enriched metadata from manifest by file path (more reliable than hash)
        enriched_meta = enriched_manifest.get(file_path, {})
        ceo_name = enriched_meta.get("ceo_name")
        ceo_title = enriched_meta.get("ceo_title")
        company = enriched_meta.get("company")
        ceo_display = enriched_meta.get("ceo_display")
        host = enriched_meta.get("host")
        date = enriched_meta.get("date")

        if skip_unchanged:
            try:
                existing = coll.get(where={"record_id": {"$eq": record_id}}, include=["metadatas", "ids"], limit=1)  # type: ignore
                metas = existing.get("metadatas") or []
                if metas and metas[0] and metas[0].get("file_hash") == file_hash:
                    console.log(f"Unchanged, skipping: {os.path.basename(file_path)}")
                    continue
            except Exception:
                pass

        # If any prior chunks exist for this record, delete them before re-adding
        try:
            coll.delete(where={"record_id": record_id})  # type: ignore
        except Exception:
            pass

        chunks = chunk_text_turn_aware(prepped, max_chars=max_chars, overlap_chars=overlap_chars)

        if not chunks:
            continue

        all_ids: List[str] = []
        all_texts: List[str] = []
        all_metas: List[Dict] = []

        for ci, chunk in enumerate(chunks):

            # Strip the speaker label (if any) to measure real content
            content_only = SPEAKER_RE.sub("", chunk).strip()

            # Skip chunks with less than 25 chars of real content
            if len(content_only) < 25:
                continue

            cid = f"{record_id}::chunk::{ci:05d}"
            meta = {
                "record_id": record_id,
                "file_hash": file_hash,
                "source_path": file_path,
                "filename": os.path.basename(file_path),
                "title": title,
                "category": md.get("category") or "",
                "guest": md.get("guest") or "",
                "chunk_index": ci,
                "num_chars": len(chunk),
                # Enriched metadata from manifest
                "ceo_name": ceo_name or "",
                "ceo_title": ceo_title or "",
                "company": company or "",
                "ceo_display": ceo_display or "",
                "host": host or "",
                "date": date or "",
            }
            all_ids.append(cid)
            all_texts.append(chunk)
            all_metas.append(meta)

        console.log(f"Embedding {len(all_texts)} chunks from {os.path.basename(file_path)}…")
        embeddings = provider_obj.embed(all_texts)

        # Upsert per-file to avoid massive memory
        coll.add(
            ids=all_ids,
            documents=all_texts,
            metadatas=all_metas,
            embeddings=embeddings,
        )
        total_chunks += len(all_texts)

    console.log(f"Indexing complete. Total chunks: {total_chunks}")


def hashlib_safe(s: str) -> str:
    import hashlib

    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


# -----------------------------
# Query + Synthesis
# -----------------------------


def build_query_client(
    persist_dir: str,
    collection: str,
):
    client = get_chroma_client(persist_dir)
    coll = get_or_create_collection(client, collection)
    return coll


def get_date_range_from_db(
    persist_dir: str = default_persist_dir(),
    collection: str = "transcripts",
) -> Tuple[Optional[str], Optional[str]]:
    """Get the minimum and maximum dates from the ChromaDB collection.
    
    Returns:
        (min_date, max_date) tuple in YYYY-MM-DD format, or (None, None) if no dates found
    """
    try:
        client = get_chroma_client(persist_dir)
        coll = get_or_create_collection(client, collection)
        
        # Get all unique dates (sample approach - get many records and find min/max)
        # ChromaDB doesn't have a direct aggregation API, so we get a large sample
        results = coll.get(
            where={"date": {"$ne": ""}},  # Only get records with non-empty dates
            limit=10000,  # Get large sample
            include=["metadatas"]
        )
        
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return None, None
        
        # Extract all valid dates
        dates = []
        for meta in metadatas:
            if meta and meta.get("date"):
                date_str = meta["date"]
                # Validate YYYY-MM-DD format
                if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                    dates.append(date_str)
        
        if not dates:
            return None, None
        
        # Return min and max
        return min(dates), max(dates)
        
    except Exception as e:
        console.log(f"[yellow]Could not get date range from DB: {e}[/yellow]")
        return None, None


def extract_keywords(question: str, max_terms: int = 8) -> List[str]:
    # Very light keyword extraction: content words only
    q = question.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    tokens = [t for t in q.split() if len(t) > 2]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "about",
        "from",
        "that",
        "this",
        "what",
        "when",
        "your",
        "into",
        "have",
        "how",
        "are",
        "can",
        "you",
        "some",
        "like",
        "give",
        "back",
        "then",
        "them",
        "will",
        "they",
        "their",
        "there",
        "over",
    }
    content = [t for t in tokens if t not in stop]
    # naive de-dup preserving order
    seen = set()
    out: List[str] = []
    for t in content:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= max_terms:
            break
    return out


def compute_text_similarity(a: str, b: str) -> float:
    """Return a normalized fuzzy similarity between two texts in [0,1]."""
    try:
        score = rf_fuzz.token_set_ratio(a, b)
        return float(score) / 100.0
    except Exception:
        return 0.0


def analyze_query_intent(
    question: str,
    model: str = "gpt-5-nano",
) -> Dict[str, any]:
    """Use LLM to deeply analyze the user's question.
    
    Extracts:
    - Key phrases and business concepts
    - Important entities (companies, roles, industries)
    - Intent and context
    - Suggested search angles
    
    Returns dict with extracted information, or empty dict if unavailable.
    """
    if OpenAI is None:
        return {}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}
    client = OpenAI(api_key=api_key)
    
    prompt = (
        "Analyze this business/leadership question to optimize search in a CEO interview database.\n\n"
        "Question: " + question + "\n\n"
        "Extract and return (in JSON format):\n"
        "1. key_phrases: List of 3-5 critical business phrases (2-5 words each) that capture the core concepts\n"
        "2. entities: List of specific entities mentioned (company names, executive roles, industries, business functions, methodologies)\n"
        "3. intent: One concise sentence describing what actionable advice or insight the user seeks\n"
        "4. business_context: The specific business domain/challenge (e.g., 'post-merger culture integration', 'scaling digital marketing teams', 'customer retention strategy', 'hiring top talent')\n"
        "5. search_angles: 2-3 alternative semantic angles to find relevant CEO advice:\n"
        "   - Use executive terminology (e.g., 'talent acquisition' vs 'hiring')\n"
        "   - Focus on outcomes, processes, and lessons learned\n"
        "   - Consider temporal aspects (e.g., 'during growth phase', 'after acquisition')\n\n"
        "Return ONLY valid JSON, no other text."
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing business questions to optimize semantic search in CEO interview content. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        )
        import json as _json
        result = _json.loads(resp.choices[0].message.content or "{}")
        return result
    except Exception as e:
        console.log(f"[yellow]Query analysis failed: {e}[/yellow]")
        return {}


def evaluate_snippet_relevance(
    question: str,
    snippet_text: str,
    model: str = "gpt-5-nano",
) -> Dict[str, any]:
    """Evaluate how relevant a snippet is to the question.
    
    Returns dict with:
    - score: 0-10 relevance score
    - reasoning: Brief explanation
    - is_relevant: Boolean (score >= 6)
    
    Returns default low score if evaluation fails.
    """
    if OpenAI is None:
        return {"score": 5, "reasoning": "Evaluation unavailable", "is_relevant": False}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"score": 5, "reasoning": "Evaluation unavailable", "is_relevant": False}
    client = OpenAI(api_key=api_key)
    
    # Use full snippet text for accurate relevance evaluation
    prompt = (
        "Rate how relevant this CEO interview excerpt is to answering the business question.\n\n"
        f"Question: {question}\n\n"
        f"Interview Excerpt: {snippet_text}\n\n"
        "Return JSON with:\n"
        "- score: Integer 0-10 (0=completely irrelevant, 10=perfectly relevant)\n"
        "- reasoning: One brief sentence explaining the score\n\n"
        "Scoring guidelines:\n"
        "• 9-10: Directly answers the question with specific, actionable CEO advice or detailed examples\n"
        "• 7-8: Addresses the topic with relevant insights but less specific or actionable detail\n"
        "• 5-6: Tangentially related to the topic but doesn't directly address the question\n"
        "• 3-4: Mentions the domain/industry but lacks relevant insights\n"
        "• 0-2: Completely unrelated or just pleasantries/filler conversation\n\n"
        "Prioritize snippets that:\n"
        "- Provide concrete strategies, processes, or frameworks\n"
        "- Share specific examples or case studies\n"
        "- Offer lessons learned or mistakes to avoid\n"
        "- Give actionable steps or recommendations\n\n"
        "Return ONLY valid JSON."
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at evaluating the relevance of CEO interview content to business questions. Focus on actionable insights, specific examples, and strategic advice. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="minimal"  # Use minimal reasoning for fast evaluation
        )
        import json as _json
        result = _json.loads(resp.choices[0].message.content or '{"score": 5, "reasoning": "Parse error"}')
        score = int(result.get("score", 5))
        return {
            "score": score,
            "reasoning": result.get("reasoning", ""),
            "is_relevant": score >= 6
        }
    except Exception as e:
        console.log(f"[yellow]Relevance evaluation failed: {e}[/yellow]")
        return {"score": 5, "reasoning": "Evaluation error", "is_relevant": False}


def extract_relevant_excerpt(
    question: str,
    full_text: str,
    model: str = "gpt-5-nano",
    max_length: int = 800,
) -> str:
    """Extract the most relevant portion of a long text for display.
    
    Uses LLM to identify and extract 2-4 complete sentences that best answer
    the question. Falls back to simple truncation if extraction fails.
    """
    # If text is already short enough, return as-is
    if len(full_text) <= max_length:
        return full_text
    
    if OpenAI is None:
        return full_text[:max_length].strip() + "..."
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return full_text[:max_length].strip() + "..."
    
    client = OpenAI(api_key=api_key)
    
    prompt = (
        "Extract the most relevant portion of this CEO interview that directly answers the question.\n\n"
        f"Question: {question}\n\n"
        f"Full Text: {full_text}\n\n"
        "Instructions:\n"
        "1. Find the 2-4 consecutive sentences that BEST answer the question\n"
        "2. Extract them as complete sentences (don't cut off mid-sentence)\n"
        "3. Keep the speaker name/label if present\n"
        "4. The excerpt should be focused and directly relevant\n"
        f"5. Aim for {max_length} characters or less\n\n"
        "Return ONLY the extracted text, nothing else."
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at extracting relevant excerpts from CEO interviews. Return only the extracted text with no additional commentary."},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="minimal"  # Use minimal reasoning for fast extraction
        )
        excerpt = (resp.choices[0].message.content or "").strip()

        # Validate extraction (allow minor formatting differences)
        if excerpt and len(excerpt) > 50:
            # If excerpt is too long, cap to max_length keeping sentence integrity
            if len(excerpt) > max_length:
                trimmed = excerpt[:max_length].rsplit(". ", 1)[0]
                return (trimmed + ".") if trimmed else excerpt[:max_length]
            return excerpt
        else:
            # Fallback to simple truncation
            return full_text[:max_length].strip() + "..."
            
    except Exception as e:
        console.log(f"[yellow]Excerpt extraction failed: {e}[/yellow]")
        return full_text[:max_length].strip() + "..."


def batch_evaluate_snippets(
    question: str,
    snippets: List[Dict],
    model: str = "gpt-5-nano",
    min_relevance: int = 6,
    max_workers: int = 10,
) -> List[Dict]:
    """Evaluate multiple snippets in parallel and return only relevant ones.
    
    Adds 'relevance_score' and 'relevance_reasoning' to each snippet.
    Filters out snippets with score < min_relevance.
    """
    if OpenAI is None:
        return snippets  # Return all if evaluation unavailable
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return snippets
    
    # Parallel evaluation for efficiency
    evaluated = []
    
    # Filter snippets with text
    valid_snippets = [(i, s) for i, s in enumerate(snippets) if s.get("text", "")]
    
    if not valid_snippets:
        return []
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation tasks
        future_to_snippet = {
            executor.submit(evaluate_snippet_relevance, question, snippet.get("text", ""), model): (idx, snippet)
            for idx, snippet in valid_snippets
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_snippet):
            idx, snippet = future_to_snippet[future]
            try:
                eval_result = future.result()
                
                # Add evaluation data to snippet
                snippet["relevance_score"] = eval_result["score"]
                snippet["relevance_reasoning"] = eval_result["reasoning"]
                
                # Only keep if meets threshold
                if eval_result["score"] >= min_relevance:
                    evaluated.append(snippet)
            except Exception as e:
                console.log(f"[yellow]Evaluation failed for snippet {idx}: {e}[/yellow]")
                # Keep snippet with default score if evaluation fails
                snippet["relevance_score"] = 5
                snippet["relevance_reasoning"] = "Evaluation failed"
                if 5 >= min_relevance:
                    evaluated.append(snippet)
    
    return evaluated


def agent_propose_queries(
    question: str,
    snippets: List[Dict],
    query_analysis: Optional[Dict] = None,
    model: str = "gpt-5-nano",
    max_suggestions: int = 4,
) -> List[str]:
    """Use a lightweight LLM call to propose refined search queries.

    Uses query analysis if provided to generate more targeted queries.
    Returns up to max_suggestions strings. If OpenAI is unavailable, returns [].
    """
    if OpenAI is None:
        return []
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    client = OpenAI(api_key=api_key)

    context = []
    for s in snippets[:5]:
        title = s.get('title', '')
        text = s.get('text', '')[:200]
        score = s.get('relevance_score', 'N/A')
        context.append(f"- [{score}/10] {title} :: {text}")
    
    # Build enhanced prompt with query analysis
    analysis_text = ""
    if query_analysis:
        business_context = query_analysis.get('business_context', '')
        key_phrases = query_analysis.get('key_phrases', [])
        if business_context:
            analysis_text = f"\nBusiness context: {business_context}\n"
        if key_phrases:
            analysis_text += f"Key phrases: {', '.join(key_phrases)}\n"
    
    prompt = (
        "Generate 2-4 improved search queries to find CEO interview content that better answers the question.\n\n"
        f"Original Question: {question}\n"
        f"{analysis_text}"
        f"\nCurrent Results (with relevance scores 0-10):\n" + "\n".join(context) + "\n\n"
        "Analysis: The current results have low relevance. Generate alternative queries that:\n\n"
        "1. **Use executive/CEO terminology and phrasing:**\n"
        "   - Replace casual words with business terminology\n"
        "   - Examples: 'hiring' → 'talent acquisition strategy', 'culture' → 'organizational culture transformation'\n\n"
        "2. **Focus on specific challenges and solutions:**\n"
        "   - Query for processes, frameworks, and methodologies\n"
        "   - Include words like: approach, strategy, framework, process, lessons, challenges, success factors\n\n"
        "3. **Target concrete examples and case studies:**\n"
        "   - Include phrases like: 'when we', 'how I', 'in my experience', 'example of'\n"
        "   - Look for stories, not just abstract advice\n\n"
        "4. **Explore different semantic angles:**\n"
        "   - What to do (strategies, best practices)\n"
        "   - What to avoid (mistakes, pitfalls, lessons learned)\n"
        "   - How to do it (implementation, execution, tactical steps)\n"
        "   - Why it matters (business impact, outcomes, results)\n\n"
        "5. **Use complete phrases (4-6 words) that appear in executive conversations:**\n"
        "   - 'building high-performing teams after merger'\n"
        "   - 'scaling digital marketing operations rapidly'\n"
        "   - 'integrating acquired company cultures successfully'\n\n"
        "6. **Consider context and timing:**\n"
        "   - Add temporal qualifiers: 'during rapid growth', 'after acquisition', 'when entering new markets'\n"
        "   - Include scale/size context: 'large organization', 'startup to enterprise', '100+ person team'\n\n"
        "Return ONLY 2-4 search query strings, one per line. No numbering, bullets, explanations, or other text."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at generating semantic search queries to find relevant CEO interview content and executive business advice. Focus on executive terminology, actionable insights, and specific business challenges. Return only queries, one per line."},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="minimal"  # Use minimal reasoning for fast query generation
        )
        text = resp.choices[0].message.content or ""
        lines = [ln.strip("- ").strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith('#')]
        return lines[:max_suggestions]
    except Exception as e:
        console.log(f"[yellow]Query generation failed: {e}[/yellow]")
        return []


def compose_answer_oai(
    question: str,
    snippets: List[Dict],
    model: str = "gpt-5-mini",
) -> str:
    if OpenAI is None:
        raise RuntimeError("openai python package not installed; cannot compose answer")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot compose answer")
    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are an expert business advisor synthesizing insights from CEO interviews to answer leadership questions. "
        "Your role is to provide comprehensive, actionable answers grounded in real executive experience.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. **Attribution Format**: ALWAYS attribute advice to specific CEOs using this exact format:\n"
        "   '**[CEO Name]** ([Title], [Company]) [action verb]...'\n"
        "   Example: '**Mary Barra** (CEO, General Motors) emphasizes that...'\n"
        "   Example: '**Mike Critelli** (Former CEO, Pitney Bowes) recommends...'\n\n"
        "2. **Structure Your Answer**:\n"
        "   - Start with a direct answer to the question\n"
        "   - Group related insights by theme or approach\n"
        "   - Use clear headers if covering multiple aspects\n"
        "   - End with key takeaways if appropriate\n\n"
        "3. **Content Quality**:\n"
        "   - Prioritize specific, actionable advice over general statements\n"
        "   - Include concrete examples, frameworks, or processes mentioned by CEOs\n"
        "   - Highlight lessons learned and mistakes to avoid\n"
        "   - When multiple CEOs address the same topic, compare or synthesize their perspectives\n\n"
        "4. **Voice and Style**:\n"
        "   - Write in a professional, advisory tone\n"
        "   - Use executive terminology appropriate to the context\n"
        "   - Be concise but comprehensive - every sentence should add value\n"
        "   - Ground ALL advice in the provided snippets - do not add external knowledge"
    )
    content_blocks = []
    for i, s in enumerate(snippets, 1):
        who = s.get("speaker_display") or s.get("ceo_display") or s.get("speaker_label") or "Unknown Speaker"
        company = s.get("company") or s.get("speaker_org") or ""
        title = s.get("title", "unknown")
        text = s.get("text", "")
        
        # Format with clear separation and numbering for better synthesis
        block = f"--- Snippet {i} ---\n"
        block += f"Source: {title}\n"
        block += f"Speaker: {who}\n"
        if company:
            block += f"Company: {company}\n"
        block += f"\nContent:\n{text}"
        content_blocks.append(block)
    
    user_content = (
        f"Business Question: {question}\n\n"
        f"Number of CEO Interview Excerpts: {len(snippets)}\n\n"
        "Your task: Synthesize these CEO insights into a comprehensive, well-structured answer.\n"
        "Remember to:\n"
        "- Attribute every insight to the specific CEO (with proper formatting)\n"
        "- Focus on actionable advice and concrete examples\n"
        "- Organize insights logically by theme or approach\n"
        "- Compare perspectives when multiple CEOs address the same aspect\n\n"
        "--- CEO INTERVIEW EXCERPTS ---\n\n" + "\n\n".join(content_blocks)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        reasoning_effort="minimal"  # Use medium reasoning for quality answer composition
    )
    return resp.choices[0].message.content or ""


def run_query(
    question: str,
    persist_dir: str = default_persist_dir(),
    collection: str = "auto",
    collections: Optional[List[str]] = None,
    top_k: int = 8,
    expand_if_low: bool = True,
    compose: bool = True,
    composition_model: str = "gpt-5-mini",
    evaluation_model: str = "gpt-5-nano",  # Separate model for intermediate evaluations
    max_snippet_chars: int = 600,
    full_turn: bool = True,
    hybrid_rerank: bool = True,
    hybrid_alpha: float = 0.7,
    agent_expand: bool = True,
    max_agent_rounds: int = 2,
    metadata_agent: bool = True,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> Dict:
    """Programmatic interface for querying the RAG index.

    Returns a structured dict with keys: question, keywords, answer, snippets, sources.
    
    Args:
        date_min: Minimum date filter (YYYY-MM-DD format), inclusive
        date_max: Maximum date filter (YYYY-MM-DD format), inclusive
    """
    import time
    t_start = time.time()
    console.log(f"[cyan]⏱️  Starting RAG query pipeline[/cyan]")
    
    question = question.strip()
    
    # Step 1: Skip query intent analysis for performance (was adding 1 LLM call + latency)
    # console.log(f"[cyan]Analyzing query...[/cyan]")
    # query_analysis = analyze_query_intent(question, model=composition_model)
    # if query_analysis:
    #     console.log(f"[cyan]Context: {query_analysis.get('business_context', 'N/A')}[/cyan]")
    query_analysis = {}  # Empty dict to maintain compatibility
    
    # Determine collections to query
    coll_names: List[str]
    if collections and len(collections) > 0:
        coll_names = list(dict.fromkeys(collections))
    elif collection and collection != "auto":
        coll_names = [collection]
    else:
        coll_names = ["transcripts", "magazines"]

    # Build clients for each collection
    t_colls = time.time()
    client = get_chroma_client(persist_dir)
    colls = []
    for name in coll_names:
        try:
            colls.append(get_or_create_collection(client, name))
        except Exception:
            # Skip if cannot open
            pass
    console.log(f"[cyan]⏱️  Collections loaded ({time.time() - t_colls:.2f}s)[/cyan]")

    # prep keywords (not used for filtering here, but can be logged and used later for hybrid)
    keywords = extract_keywords(question)

    # Retrieve more candidates initially (2x top_k) for better filtering pool
    # The LLM relevance evaluation will filter down to the best ones
    n_results = max(top_k * 2, 10)  # Reduced from 3x to 2x for performance
    
    # Always embed the query with OpenAI text-embedding-3-small to match the index
    t_embed = time.time()
    q_provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small")
    q_emb = q_provider.embed([question])
    console.log(f"[cyan]⏱️  Query embedding ({time.time() - t_embed:.2f}s)[/cyan]")
    
    # Build ChromaDB where clause for date filtering
    # IMPORTANT: where_clause remains None when no date filter is active
    # This ensures ALL records (including those without dates) are returned when filter is disabled
    where_clause: Optional[Dict] = None
    if date_min or date_max:
        conditions = []
        
        # Date is stored as string in YYYY-MM-DD format, empty string if missing
        # ChromaDB supports $gte and $lte operators for string comparison
        # When filtering IS active, we exclude records with empty dates since they can't be in range
        if date_min and date_max:
            # Both bounds: date >= date_min AND date <= date_max
            where_clause = {
                "$and": [
                    {"date": {"$gte": date_min}},
                    {"date": {"$lte": date_max}},
                    {"date": {"$ne": ""}}  # Exclude empty dates
                ]
            }
            console.log(f"[cyan]📅 Date filter: {date_min} to {date_max}[/cyan]")
        elif date_min:
            # Only minimum: date >= date_min
            where_clause = {
                "$and": [
                    {"date": {"$gte": date_min}},
                    {"date": {"$ne": ""}}
                ]
            }
            console.log(f"[cyan]📅 Date filter: from {date_min}[/cyan]")
        elif date_max:
            # Only maximum: date <= date_max
            where_clause = {
                "$and": [
                    {"date": {"$lte": date_max}},
                    {"date": {"$ne": ""}}
                ]
            }
            console.log(f"[cyan]📅 Date filter: until {date_max}[/cyan]")
    
    # Query all collections in parallel and merge initial results
    t_retrieval = time.time()
    all_results = []
    if len(colls) > 1:
        # Parallel querying for multiple collections
        def query_collection(coll):
            try:
                if where_clause:
                    return coll.query(query_embeddings=q_emb, n_results=n_results, where=where_clause, include=["metadatas", "documents", "distances"])  # type: ignore
                else:
                    return coll.query(query_embeddings=q_emb, n_results=n_results, include=["metadatas", "documents", "distances"])  # type: ignore
            except Exception:
                return None
        
        with ThreadPoolExecutor(max_workers=len(colls)) as executor:
            future_to_coll = {executor.submit(query_collection, coll): coll for coll in colls}
            for future in as_completed(future_to_coll):
                res = future.result()
                if res is not None:
                    all_results.append(res)
    else:
        # Single collection - no need for parallelization overhead
        for coll in colls:
            try:
                if where_clause:
                    res = coll.query(query_embeddings=q_emb, n_results=n_results, where=where_clause, include=["metadatas", "documents", "distances"])  # type: ignore
                else:
                    res = coll.query(query_embeddings=q_emb, n_results=n_results, include=["metadatas", "documents", "distances"])  # type: ignore
                all_results.append(res)
            except Exception:
                continue
    console.log(f"[cyan]⏱️  Initial retrieval ({time.time() - t_retrieval:.2f}s, {len(all_results)} collections)[/cyan]")

    # Heuristic expansion based on best distance across all initial results
    def best_distance(res_list: List[Dict]) -> float:
        best_d = 1.0
        for r in res_list:
            ds = r.get("distances", [[1.0]])[0]
            if ds:
                best_d = min(best_d, ds[0])
        return best_d

    try:
        best = best_distance(all_results)
        # If still not good results, expand even further
        if expand_if_low and best > 0.35 and n_results < 20:
            n_results = 20
            all_results = []
            
            if len(colls) > 1:
                # Parallel expansion query
                def query_collection_expanded(coll):
                    try:
                        if where_clause:
                            return coll.query(query_embeddings=q_emb, n_results=n_results, where=where_clause, include=["metadatas", "documents", "distances"])  # type: ignore
                        else:
                            return coll.query(query_embeddings=q_emb, n_results=n_results, include=["metadatas", "documents", "distances"])  # type: ignore
                    except Exception:
                        return None
                
                with ThreadPoolExecutor(max_workers=len(colls)) as executor:
                    future_to_coll = {executor.submit(query_collection_expanded, coll): coll for coll in colls}
                    for future in as_completed(future_to_coll):
                        res = future.result()
                        if res is not None:
                            all_results.append(res)
            else:
                # Single collection
                for coll in colls:
                    try:
                        if where_clause:
                            res = coll.query(query_embeddings=q_emb, n_results=n_results, where=where_clause, include=["metadatas", "documents", "distances"])  # type: ignore
                        else:
                            res = coll.query(query_embeddings=q_emb, n_results=n_results, include=["metadatas", "documents", "distances"])  # type: ignore
                        all_results.append(res)
                    except Exception:
                        continue
    except Exception:
        pass

    # Flatten results
    flat_docs: List[str] = []
    flat_metas: List[Dict] = []
    flat_dists: List[float] = []
    for r in all_results:
        docs = r.get("documents", [[]])[0]
        metas = r.get("metadatas", [[]])[0]
        dists = r.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            flat_docs.append(doc)
            flat_metas.append(meta)
            flat_dists.append(dist)

    # Build candidate list (before possible rerank/agent expansion)
    t_candidates = time.time()
    candidates: List[Dict] = []
    
    # Extract entities from query analysis for metadata boosting
    query_entities = []
    if query_analysis:
        query_entities = [e.lower() for e in query_analysis.get("entities", [])]
    
    for doc, meta, dist in zip(flat_docs, flat_metas, flat_dists):
        snippet_text = doc[:max_snippet_chars].strip()
        filename = meta.get("filename") or (os.path.basename(meta.get("source_path", "")) if meta.get("source_path") else None)
        category = meta.get("category")
        title = meta.get("title", "")
        
        # build synthetic id
        rid = meta.get("record_id") or "rec"
        try:
            ci = int(meta.get("chunk_index", -1))
        except Exception:
            ci = -1
        sid = f"{rid}::chunk::{ci:05d}" if ci >= 0 else rid
        emb_score = float(1.0 - dist)
        txt_sim = compute_text_similarity(question, doc)
        hybrid_score = hybrid_alpha * emb_score + (1.0 - hybrid_alpha) * txt_sim if hybrid_rerank else emb_score
        
        # Apply metadata boost if title/filename matches query entities
        metadata_boost = 0.0
        if query_entities and (filename or title):
            search_text = f"{filename or ''} {title}".lower()
            for entity in query_entities:
                if entity in search_text:
                    metadata_boost = 0.1  # 10% boost for metadata match
                    break
        
        # Apply boost to hybrid score
        boosted_score = min(1.0, hybrid_score + metadata_boost)
        
        candidates.append(
            {
                "id": sid,
                "emb_score": emb_score,
                "txt_score": txt_sim,
                "hybrid_score": boosted_score,
                "doc": doc,
                "meta": meta,
                "filename": filename,
                "category": category,
                "record_id": rid,
            }
        )
    console.log(f"[cyan]⏱️  Built {len(candidates)} candidates ({time.time() - t_candidates:.2f}s)[/cyan]")

    # Adaptive retrieval with relevance evaluation (limited to single round for performance)
    t_evaluation = time.time()
    retrieval_rounds = 0
    max_retrieval_rounds = 1  # Always 1 round for performance (was: max_agent_rounds if agent_expand else 1)
    min_avg_relevance = 6.0  # Lowered threshold to accept results faster (was 7.0)
    
    while retrieval_rounds < max_retrieval_rounds:
        retrieval_rounds += 1
        
        # Sort by hybrid score and take top candidates for evaluation
        key_name = "hybrid_score" if hybrid_rerank else "emb_score"
        candidates.sort(key=lambda c: c[key_name], reverse=True)
        top_candidates = candidates[:min(top_k, len(candidates))]  # Reduced from top_k * 2 for performance
        
        # Build provisional snippets for evaluation (use FULL text for accurate scoring)
        provisional_snippets = [
            {
                "title": c["meta"].get("title"),
                "text": c["doc"],  # Full document text for relevance evaluation
            }
            for c in top_candidates
        ]
        
        # Evaluate relevance of top candidates
        console.log(f"[cyan]Evaluating {len(provisional_snippets)} snippets... (round {retrieval_rounds})[/cyan]")
        evaluated_snippets = batch_evaluate_snippets(
            question, 
            provisional_snippets, 
            model=evaluation_model,  # Use nano for fast evaluation
            min_relevance=5  # Low threshold for initial pass
        )
        
        # Calculate average relevance
        if evaluated_snippets:
            avg_relevance = sum(s.get("relevance_score", 0) for s in evaluated_snippets) / len(evaluated_snippets)
            console.log(f"[cyan]Average relevance: {avg_relevance:.1f}/10[/cyan]")
            
            # Update candidates with relevance scores
            for i, candidate in enumerate(top_candidates):
                if i < len(evaluated_snippets):
                    candidate["relevance_score"] = evaluated_snippets[i].get("relevance_score", 5)
                    candidate["relevance_reasoning"] = evaluated_snippets[i].get("relevance_reasoning", "")
        else:
            avg_relevance = 5.0
            console.log(f"[yellow]No snippets passed initial relevance filter[/yellow]")
        
        # If relevance is good enough or this is the last round, stop
        if avg_relevance >= min_avg_relevance or retrieval_rounds >= max_retrieval_rounds:
            console.log(f"[green]Retrieval complete (relevance: {avg_relevance:.1f}/10)[/green]")
            break
        
        # Generate improved queries for next round
        console.log(f"[yellow]Low relevance detected. Generating improved queries...[/yellow]")
        suggestions = agent_propose_queries(
            question, 
            evaluated_snippets, 
            query_analysis=query_analysis,
            model=evaluation_model  # Use nano for fast query generation
        )
        
        if not suggestions:
            console.log(f"[yellow]No query suggestions generated. Stopping retrieval.[/yellow]")
            break
        
        console.log(f"[cyan]Trying {len(suggestions)} alternative queries in parallel...[/cyan]")
        
        # Helper function to process a single alternative query
        def process_alternative_query(q2: str):
            new_candidates = []
            try:
                q2_emb = q_provider.embed([q2])
                for coll in colls:
                    try:
                        if where_clause:
                            r2 = coll.query(query_embeddings=q2_emb, n_results=n_results, where=where_clause, include=["metadatas", "documents", "distances"])  # type: ignore
                        else:
                            r2 = coll.query(query_embeddings=q2_emb, n_results=n_results, include=["metadatas", "documents", "distances"])  # type: ignore
                        d2 = r2.get("documents", [[]])[0]
                        m2 = r2.get("metadatas", [[]])[0]
                        dist2 = r2.get("distances", [[]])[0]
                        for doc2, meta2, distv in zip(d2, m2, dist2):
                            rid2 = meta2.get("record_id") or "rec"
                            try:
                                ci2 = int(meta2.get("chunk_index", -1))
                            except Exception:
                                ci2 = -1
                            sid2 = f"{rid2}::chunk::{ci2:05d}" if ci2 >= 0 else rid2
                            filename2 = meta2.get("filename") or (os.path.basename(meta2.get("source_path", "")) if meta2.get("source_path") else None)
                            category2 = meta2.get("category")
                            title2 = meta2.get("title", "")
                            emb_score2 = float(1.0 - distv)
                            txt_sim2 = compute_text_similarity(question, doc2)
                            hybrid_score2 = hybrid_alpha * emb_score2 + (1.0 - hybrid_alpha) * txt_sim2 if hybrid_rerank else emb_score2
                            
                            # Apply metadata boost for expanded results too
                            metadata_boost2 = 0.0
                            if query_entities and (filename2 or title2):
                                search_text2 = f"{filename2 or ''} {title2}".lower()
                                for entity in query_entities:
                                    if entity in search_text2:
                                        metadata_boost2 = 0.1
                                        break
                            
                            boosted_score2 = min(1.0, hybrid_score2 + metadata_boost2)
                            
                            new_candidates.append(
                                {
                                    "id": sid2,
                                    "emb_score": emb_score2,
                                    "txt_score": txt_sim2,
                                    "hybrid_score": boosted_score2,
                                    "doc": doc2,
                                    "meta": meta2,
                                    "filename": filename2,
                                    "category": category2,
                                    "record_id": rid2,
                                }
                            )
                    except Exception:
                        continue
            except Exception:
                pass
            return new_candidates
        
        # Process alternative queries in parallel
        with ThreadPoolExecutor(max_workers=len(suggestions)) as executor:
            future_to_query = {executor.submit(process_alternative_query, q2): q2 for q2 in suggestions}
            
            for future in as_completed(future_to_query):
                try:
                    new_candidates = future.result()
                    # Add only unique candidates
                    for cand in new_candidates:
                        if not any(c["id"] == cand["id"] for c in candidates):
                            candidates.append(cand)
                except Exception as e:
                    console.log(f"[yellow]Alternative query failed: {e}[/yellow]")
    console.log(f"[cyan]⏱️  Relevance evaluation complete ({time.time() - t_evaluation:.2f}s)[/cyan]")

    # Sort candidates by relevance score if available, otherwise by hybrid/embedding score
    # Relevance score is 0-10, normalize to 0-1 for combination
    def get_final_score(candidate: Dict) -> float:
        key_name = "hybrid_score" if hybrid_rerank else "emb_score"
        base_score = candidate.get(key_name, 0.0)
        
        # If we have a relevance score from LLM evaluation, blend it with embedding score
        if "relevance_score" in candidate:
            relevance_norm = candidate["relevance_score"] / 10.0
            # 60% LLM relevance, 40% embedding similarity
            return 0.6 * relevance_norm + 0.4 * base_score
        return base_score
    
    candidates.sort(key=get_final_score, reverse=True)

    # Build final snippets from candidates with intelligent expansion/skip to fulfill top_k
    t_snippets = time.time()
    snippets: List[Dict] = []
    def decide_expansion(
        q: str,
        cur_text: str,
        prev_turn: Optional[str],
        next_turn: Optional[str],
        speaker_map: Dict[str, Dict[str, Optional[str]]],
        cur_label: Optional[str],
    ) -> str:
        """Return one of: 'as_is', 'with_prev', 'with_next', 'skip'."""
        cur_is_question = _is_question(cur_text)
        cur_is_filler = _is_filler_turn(cur_text)
        prev_is_filler = _is_filler_turn(prev_turn or "") if prev_turn else False
        next_is_filler = _is_filler_turn(next_turn or "") if next_turn else False

        host_labels = set(_detect_host_labels(speaker_map))
        cur_key = normalize_speaker_key(cur_label or extract_speaker_label(cur_text) or "") or ""
        prev_key = normalize_speaker_key(extract_speaker_label(prev_turn or "") or "") or ""
        next_key = normalize_speaker_key(extract_speaker_label(next_turn or "") or "") or ""

        # If current is clearly filler, prefer to continue or look back; otherwise skip
        if cur_is_filler:
            if next_turn and not next_is_filler:
                return "with_next"
            if prev_turn and not prev_is_filler:
                return "with_prev"
            return "skip"

        # If current is likely a host question, continue to the guest answer
        if cur_is_question or (cur_key and cur_key in host_labels):
            if next_turn and (not next_key or next_key not in host_labels):
                return "with_next"
            # fall back to as_is if no clear guest next
            return "as_is"

        # If previous looks like the question and we have the answer now, include with prev
        if prev_turn and (_is_question(prev_turn) or (prev_key and prev_key in host_labels)):
            return "with_prev"

        # Default: include as is
        return "as_is"

    for c in candidates:
        meta = c["meta"]
        doc = c["doc"]
        filename = c["filename"]
        category = c["category"]
        display_text = doc[:max_snippet_chars].strip()
        full_text = doc
        speaker_label: Optional[str] = None
        prev_turn: Optional[str] = None
        next_turn: Optional[str] = None
        if full_turn and meta.get("source_path"):
            expanded_text, speaker_label, prev_turn, next_turn = expand_chunk_to_full_turn(doc, meta["source_path"])  # use full doc text for matching
            display_text = expanded_text[:max_snippet_chars].strip()
            full_text = expanded_text

        # Metadata agent: Use only pre-stored enriched metadata (no runtime inference)
        speaker_display: Optional[str] = None
        speaker_name: Optional[str] = None
        speaker_role: Optional[str] = None
        speaker_org: Optional[str] = None
        sp_map: Dict = {}
        
        # Use enriched metadata from the index (already populated during indexing)
        ceo_display_stored = meta.get("ceo_display")
        ceo_name_stored = meta.get("ceo_name")
        ceo_title_stored = meta.get("ceo_title")
        company_stored = meta.get("company")
        
        if ceo_display_stored and ceo_name_stored:
            # Use pre-enriched metadata - much faster and more consistent
            speaker_name = ceo_name_stored
            speaker_role = ceo_title_stored
            speaker_org = company_stored
            speaker_display = ceo_display_stored
        # Note: Runtime inference removed for performance - all metadata should be pre-indexed

        try:
            ci3 = int(meta.get("chunk_index", -1))
        except Exception:
            ci3 = -1

        # Decide whether to include as-is, include with neighbor, or skip
        decision = decide_expansion(
            question,
            full_text,
            prev_turn,
            next_turn,
            sp_map,
            speaker_label,
        ) if full_turn else "as_is"

        if decision == "skip":
            continue

        if decision == "with_next" and next_turn:
            combined = full_text + "\n\n" + next_turn
            full_text_out = combined
            display_text_out = combined[:max_snippet_chars].strip()
        elif decision == "with_prev" and prev_turn:
            combined = prev_turn + "\n\n" + full_text
            full_text_out = combined
            display_text_out = combined[:max_snippet_chars].strip()
        else:
            full_text_out = full_text
            display_text_out = display_text

        # Include relevance information if available
        relevance_score = c.get("relevance_score")
        relevance_reasoning = c.get("relevance_reasoning")
        
        # Skip snippets with low relevance scores (if evaluated)
        min_relevance_threshold = 6
        if relevance_score is not None and relevance_score < min_relevance_threshold:
            continue
        
        snippet_dict = {
            "id": c["id"],
            "score": float(c[key_name]),
            "emb_score": c["emb_score"],
            "txt_score": c["txt_score"],
            "text": display_text_out,
            "full_text": full_text_out if full_turn else None,
            "expansion_decision": decision,
            "speaker_label": speaker_label,
            "speaker_name": speaker_name,
            "speaker_role": speaker_role,
            "speaker_org": speaker_org,
            "speaker_display": speaker_display,
            "company": company_stored,  # Add company to snippet
            "title": meta.get("title"),
            "category": category,
            "guest": meta.get("guest"),
            "date": meta.get("date"),  # Add date for filtering
            "source_path": meta.get("source_path"),
            "filename": filename,
            "source": filename,
            "source_type": category,
            "record_id": meta.get("record_id"),
            "chunk_index": meta.get("chunk_index"),
        }
        
        # Add relevance info if available
        if relevance_score is not None:
            snippet_dict["relevance_score"] = relevance_score
            snippet_dict["relevance_reasoning"] = relevance_reasoning
        
        snippets.append(snippet_dict)

        if len(snippets) >= top_k:
            break
    console.log(f"[cyan]⏱️  Built {len(snippets)} final snippets ({time.time() - t_snippets:.2f}s)[/cyan]")

    answer: Optional[str] = None
    if compose and snippets:
        t_compose = time.time()
        try:
            answer = compose_answer_oai(question, snippets[:top_k], model=composition_model)
            # Post-process to remove a leading 'Direct answer' heading if present
            answer = strip_leading_direct_answer(answer)
            console.log(f"[cyan]⏱️  Answer composition ({time.time() - t_compose:.2f}s)[/cyan]")
        except Exception as e:
            answer = f"[Composition failed: {e}]"
            console.log(f"[red]⏱️  Answer composition failed ({time.time() - t_compose:.2f}s)[/red]")

    # Aggregate per-record sources
    sources_map: Dict[str, Dict] = {}
    for s in snippets:
        rid = s.get("record_id") or ""
        if not rid:
            continue
        entry = sources_map.get(rid)
        base = {
            "record_id": rid,
            "title": s.get("title"),
            "category": s.get("category"),
            "guest": s.get("guest"),
            "date": s.get("date"),
            "filename": s.get("filename"),
            "source_path": s.get("source_path"),
            "best_score": s.get("score", 0.0),
            "num_snippets": 1,
        }
        if entry is None:
            sources_map[rid] = base
        else:
            entry["num_snippets"] = int(entry.get("num_snippets", 1)) + 1
            if float(s.get("score", 0.0)) > float(entry.get("best_score", 0.0)):
                entry["best_score"] = s.get("score", 0.0)

    response = {
        "question": question,
        "keywords": keywords,
        "answer": answer,
        "snippets": snippets[:top_k],
        "sources": list(sources_map.values()),
    }

    console.log(f"[green]✅ RAG pipeline complete: {time.time() - t_start:.2f}s total[/green]")
    return response


@app.command()
def query(
    question: str = typer.Argument(..., help="User question."),
    persist_dir: str = typer.Option(default_persist_dir(), help="Chroma persistence directory."),
    collection: str = typer.Option("auto", help="Chroma collection name (or 'auto' to query transcripts+magazines)."),
    collections: List[str] = typer.Option(None, help="Optional list of collections to query and merge."),
    top_k: int = typer.Option(5, help="Number of snippets to return (after merge/rerank)."),
    expand_if_low: bool = typer.Option(True, help="Auto-expand search if scores are weak."),
    compose: bool = typer.Option(True, help="Compose an answer using OpenAI."),
    composition_model: str = typer.Option("gpt-5-mini", help="OpenAI chat model for answer synthesis."),
    evaluation_model: str = typer.Option("gpt-5-nano", help="OpenAI model for intermediate evaluations."),
    max_snippet_chars: int = typer.Option(600, help="Trim snippets to this many characters for display."),
    json_output: bool = typer.Option(True, help="Print structured JSON response."),
    full_turn: bool = typer.Option(True, help="Expand each returned snippet to the full speaker turn."),
    hybrid_rerank: bool = typer.Option(True, help="Combine embedding score with fuzzy text similarity."),
    hybrid_alpha: float = typer.Option(0.7, help="Weight for embedding score in hybrid [0-1]."),
    agent_expand: bool = typer.Option(True, help="Use an agent to propose follow-up queries and merge results."),
    max_agent_rounds: int = typer.Option(1, help="Max agent expansion rounds."),
    metadata_agent: bool = typer.Option(True, help="Infer and attach real speaker names/roles to snippets."),
    date_min: Optional[str] = typer.Option(None, help="Minimum date filter (YYYY-MM-DD format), inclusive."),
    date_max: Optional[str] = typer.Option(None, help="Maximum date filter (YYYY-MM-DD format), inclusive."),
):
    response = run_query(
        question=question,
        persist_dir=persist_dir,
        collection=collection,
        collections=collections,
        top_k=top_k,
        expand_if_low=expand_if_low,
        compose=compose,
        composition_model=composition_model,
        evaluation_model=evaluation_model,
        max_snippet_chars=max_snippet_chars,
        full_turn=full_turn,
        hybrid_rerank=hybrid_rerank,
        hybrid_alpha=hybrid_alpha,
        agent_expand=agent_expand,
        max_agent_rounds=max_agent_rounds,
        metadata_agent=metadata_agent,
        date_min=date_min,
        date_max=date_max,
    )

    if json_output:
        sys.stdout.write(orjson.dumps(response, option=orjson.OPT_INDENT_2).decode("utf-8") + "\n")
    else:
        if response.get("answer"):
            rprint("[bold green]Answer[/bold green]")
            rprint(response["answer"])
        table = Table(title="Snippets")
        table.add_column("Score", justify="right")
        table.add_column("Title")
        table.add_column("Category")
        table.add_column("Guest")
        table.add_column("Date")
        table.add_column("Snippet")
        for s in response.get("snippets", [])[:top_k]:
            table.add_row(
                f"{s['score']:.3f}", 
                s.get("title") or "", 
                s.get("category") or "", 
                s.get("guest") or "", 
                s.get("date") or "",
                (s.get("text") or "")[:100]  # Truncate snippet for readability
            )
        console.print(table)


@app.command()
def rebuild(
    persist_dir: str = typer.Option(default_persist_dir(), help="Chroma persistence directory."),
    collection: str = typer.Option("transcripts", help="Chroma collection name."),
):
    client = get_chroma_client(persist_dir)
    try:
        client.delete_collection(collection)
        rprint(f"[bold yellow]Deleted collection[/bold yellow]: {collection}")
    except Exception:
        rprint(f"[bold red]Collection not found[/bold red]: {collection}")


if __name__ == "__main__":
    app()


