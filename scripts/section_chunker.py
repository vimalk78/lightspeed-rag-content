"""Section-aware markdown chunker with hierarchy preservation.

Walks a markdown document, splits on heading boundaries, and produces
chunks that:
- Respect section boundaries (never splits mid-section unless too large)
- Prepend ancestor headings as breadcrumb context
- Skip heading-only sections (they become breadcrumbs for children)
- Sub-split large sections while preserving breadcrumb
- Stay within the embedding model's token limit
"""

import re
from dataclasses import dataclass, field


@dataclass
class Section:
    """A markdown section with its heading and content."""
    level: int                          # heading level (1 for #, 2 for ##, etc.)
    heading: str                        # the heading text (without # prefix)
    content_lines: list[str] = field(default_factory=list)  # lines after heading, before next heading

    @property
    def content(self) -> str:
        return "\n".join(self.content_lines).strip()

    @property
    def full_text(self) -> str:
        """Heading + content."""
        prefix = "#" * self.level + " " + self.heading
        if self.content:
            return prefix + "\n\n" + self.content
        return prefix


def parse_sections(text: str) -> list[Section]:
    """Parse markdown text into a flat list of sections."""
    sections = []
    current = None
    in_code_block = False

    for line in text.split("\n"):
        # Track code blocks to avoid parsing headings inside them
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            if current:
                current.content_lines.append(line)
            continue

        if not in_code_block:
            match = re.match(r'^(#+)\s+(.*)', line)
            if match:
                # New heading found — save previous section
                if current:
                    sections.append(current)
                level = len(match.group(1))
                heading = match.group(2).strip()
                current = Section(level=level, heading=heading)
                continue

        # Regular content line
        if current:
            current.content_lines.append(line)
        # else: content before any heading — rare, skip or handle as level 0

    # Don't forget the last section
    if current:
        sections.append(current)

    return sections


def _word_count(text: str) -> int:
    return len(text.split())


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3."""
    return int(len(text.split()) * 1.3)


def _split_text(text: str, max_tokens: int) -> list[str]:
    """Split text into pieces that fit within max_tokens.

    Splits on paragraph boundaries (double newline), falls back to
    sentence boundaries, then word boundaries.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    # Try splitting on paragraphs
    paragraphs = re.split(r'\n\n+', text)
    if len(paragraphs) > 1:
        return _merge_splits(paragraphs, max_tokens, "\n\n")

    # Try splitting on sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        return _merge_splits(sentences, max_tokens, " ")

    # Last resort: split on words
    words = text.split()
    max_words = int(max_tokens / 1.3)
    parts = []
    for i in range(0, len(words), max_words):
        parts.append(" ".join(words[i:i + max_words]))
    return parts


def _merge_splits(pieces: list[str], max_tokens: int, separator: str) -> list[str]:
    """Merge small pieces back together up to max_tokens."""
    result = []
    current = ""
    for piece in pieces:
        # If a single piece exceeds max_tokens, sub-split it further
        if _estimate_tokens(piece) > max_tokens:
            if current:
                result.append(current)
                current = ""
            result.extend(_split_text(piece, max_tokens))
            continue

        candidate = current + separator + piece if current else piece
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                result.append(current)
            current = piece
    if current:
        result.append(current)
    return result


def chunk_document(text: str, max_tokens: int = 380, min_words: int = 50) -> list[dict]:
    """Chunk a markdown document with section awareness and hierarchy preservation.

    Args:
        text: The full markdown document text.
        max_tokens: Maximum tokens per chunk (should match embedding model limit).
        min_words: Minimum words for a chunk. Sections smaller than this
                   get their heading used as breadcrumb for children instead.

    Returns:
        List of dicts with 'text' and 'breadcrumb' keys.
    """
    sections = parse_sections(text)
    if not sections:
        # No headings at all — treat as one chunk
        if text.strip():
            return [{"text": text.strip(), "breadcrumb": ""}]
        return []

    chunks = []
    # heading_stack: list of (level, heading_text) for building breadcrumbs
    heading_stack: list[tuple[int, str]] = []

    i = 0
    while i < len(sections):
        section = sections[i]

        # Update heading stack — pop anything at same or deeper level
        while heading_stack and heading_stack[-1][0] >= section.level:
            heading_stack.pop()

        has_content = bool(section.content.strip())

        # Check if next section is a direct child (deeper level)
        has_child = (i + 1 < len(sections) and sections[i + 1].level > section.level)

        if not has_content and has_child:
            # Heading-only section with children — becomes breadcrumb, not a chunk
            heading_stack.append((section.level, section.heading))
            i += 1
            continue

        # Build breadcrumb from ancestor headings
        breadcrumb = " > ".join(
            "#" * level + " " + h for level, h in heading_stack
        )

        # Build chunk text
        chunk_text = section.full_text

        # If chunk fits within token limit, emit it
        if _estimate_tokens(chunk_text) <= max_tokens:
            # Check if it's too small and we can merge with next sibling
            if _word_count(chunk_text) < min_words and i + 1 < len(sections):
                next_sec = sections[i + 1]
                # Only merge with next section at same or deeper level (sibling or child)
                if next_sec.level >= section.level:
                    combined = chunk_text + "\n\n" + next_sec.full_text
                    if _estimate_tokens(combined) <= max_tokens:
                        chunks.append({
                            "text": combined,
                            "breadcrumb": breadcrumb,
                        })
                        # Add current heading to stack before skipping
                        heading_stack.append((section.level, section.heading))
                        # Skip the next section since we merged it
                        i += 2
                        continue

            chunks.append({
                "text": chunk_text,
                "breadcrumb": breadcrumb,
            })
        else:
            # Section too large — sub-split the content, keep heading as prefix
            heading_line = "#" * section.level + " " + section.heading
            breadcrumb_prefix = ""
            if breadcrumb:
                breadcrumb_prefix = breadcrumb + "\n"

            # Reserve tokens for the heading + breadcrumb
            overhead = _estimate_tokens(heading_line) + _estimate_tokens(breadcrumb_prefix)
            content_budget = max_tokens - overhead

            sub_parts = _split_text(section.content, content_budget)
            for part in sub_parts:
                chunk_text = heading_line + "\n\n" + part
                chunks.append({
                    "text": chunk_text,
                    "breadcrumb": breadcrumb,
                })

        heading_stack.append((section.level, section.heading))
        i += 1

    # Post-process: merge small chunks into their neighbor
    merged = []
    for chunk in chunks:
        if (merged
                and _word_count(chunk["text"]) < min_words
                and _estimate_tokens(merged[-1]["text"] + "\n\n" + chunk["text"]) <= max_tokens):
            # Merge into previous chunk
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + chunk["text"]
        elif (merged
                and _word_count(merged[-1]["text"]) < min_words
                and _estimate_tokens(merged[-1]["text"] + "\n\n" + chunk["text"]) <= max_tokens):
            # Previous chunk was small, merge current into it
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + chunk["text"]
        else:
            merged.append(chunk)

    return merged


def chunks_to_text_nodes(chunks: list[dict], base_metadata: dict) -> list:
    """Convert chunk dicts to llama-index TextNode objects.

    Args:
        chunks: Output from chunk_document().
        base_metadata: Metadata to attach to each node (docs_url, title, etc.)

    Returns:
        List of TextNode objects.
    """
    from llama_index.core.schema import TextNode

    nodes = []
    for chunk in chunks:
        text = chunk["text"]
        # Prepend breadcrumb as first line if present
        if chunk["breadcrumb"]:
            text = "[" + chunk["breadcrumb"] + "]\n" + text

        metadata = dict(base_metadata)
        metadata["breadcrumb"] = chunk["breadcrumb"]

        node = TextNode(text=text, metadata=metadata)
        nodes.append(node)

    return nodes
