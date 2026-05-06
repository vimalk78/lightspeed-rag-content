"""Utility script to generate embeddings."""

import argparse
import json
import os
import time
from typing import Callable, Dict

import re

import faiss
import requests
from tqdm import tqdm
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms.utils import resolve_llm

from llama_index.core.schema import TextNode
from section_chunker import chunk_document, chunks_to_text_nodes
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file.flat.base import FlatReader
from llama_index.vector_stores.faiss import FaissVectorStore

OCP_DOCS_ROOT_URL = "https://docs.openshift.com/container-platform/"
OCP_DOCS_VERSION = "4.16"
UNREACHABLE_DOCS: int = 0
RUNBOOKS_ROOT_URL = "https://github.com/openshift/runbooks/blob/master/alerts"
HERMETIC_BUILD = False


def ping_url(url: str) -> bool:
    """Check if the URL parameter is live."""
    try:
        response = requests.get(url, timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_file_title(file_path: str) -> str:
    """Extract title from the plaintext doc file."""
    title = ""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            title = file.readline().rstrip("\n").lstrip("# ")
    except Exception:  # noqa: S110
        pass
    return title


def file_metadata_func(file_path: str, docs_url_func: Callable[[str], str]) -> Dict:
    """Populate the docs_url and title metadata elements with docs URL and the page's title.

    Args:
        file_path: str: file path in str
        docs_url_func: Callable[[str], str]: lambda for the docs_url
    """
    docs_url = docs_url_func(file_path)
    title = get_file_title(file_path)
    msg = f"file_path: {file_path}, title: {title}, docs_url: {docs_url}"
    if not HERMETIC_BUILD:
        if not ping_url(docs_url):
            global UNREACHABLE_DOCS
            UNREACHABLE_DOCS += 1
            msg += ", UNREACHABLE"
    print(msg)
    return {"docs_url": docs_url, "title": title}


def ocp_file_metadata_func(file_path: str) -> Dict:
    """Populate metadata for an OCP docs page.

    Args:
        file_path: str: file path in str
    """
    docs_url = lambda file_path: (  # noqa: E731
        OCP_DOCS_ROOT_URL
        + OCP_DOCS_VERSION
        + file_path.removeprefix(EMBEDDINGS_ROOT_DIR).removesuffix("txt")
        + "html"
    )
    return file_metadata_func(file_path, docs_url)


def runbook_file_metadata_func(file_path: str) -> Dict:
    """Populate metadata for a runbook page.

    Args:
        file_path: str: file path in str
    """
    docs_url = lambda file_path: (  # noqa: E731
        RUNBOOKS_ROOT_URL + file_path.removeprefix(RUNBOOKS_ROOT_DIR)
    )
    return file_metadata_func(file_path, docs_url)


def got_whitespace(text: str) -> bool:
    """Indicate if the parameter string contains whitespace."""
    for c in text:
        if c.isspace():
            return True
    return False


def str2bool(value: str | bool) -> bool:
    """Parse CLI boolean; argparse's type=bool is wrong (bool('False') is True)."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("", "0", "n", "no", "f", "false", "off"):
        return False
    if s in ("1", "y", "yes", "t", "true", "on"):
        return True
    raise argparse.ArgumentTypeError(f"expected a boolean string, got {value!r}")


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description="embedding cli for task execution")
    parser.add_argument("-f", "--folder", help="Plain text folder path")
    parser.add_argument("-r", "--runbooks", help="Runbooks folder path")
    parser.add_argument(
        "-md",
        "--model-dir",
        default="embeddings_model",
        help="Directory containing the embedding model",
    )
    parser.add_argument(
        "-mn",
        "--model-name",
        help="HF repo id of the embedding model",
    )
    parser.add_argument(
        "-c", "--chunk", type=int, default=380, help="Chunk size for embedding"
    )
    parser.add_argument(
        "-l", "--overlap", type=int, default=0, help="Chunk overlap for embedding"
    )
    parser.add_argument(
        "-em",
        "--exclude-metadata",
        nargs="+",
        default=None,
        help="Metadata to be excluded during embedding",
    )
    parser.add_argument("-o", "--output", help="Vector DB output folder")
    parser.add_argument("-i", "--index", help="Product index")
    parser.add_argument("-v", "--ocp-version", help="OCP version")
    parser.add_argument(
        "-hb",
        "--hermetic-build",
        type=str2bool,
        default=False,
        help="Hermetic build (true/false, yes/no, 1/0)",
    )
    args = parser.parse_args()
    print(f"Arguments used: {args}")

    # OLS-823: sanitize directory
    PERSIST_FOLDER = os.path.normpath("/" + args.output).lstrip("/")
    if PERSIST_FOLDER == "":
        PERSIST_FOLDER = "."

    EMBEDDINGS_ROOT_DIR = os.path.abspath(args.folder)
    if EMBEDDINGS_ROOT_DIR.endswith("/"):
        EMBEDDINGS_ROOT_DIR = EMBEDDINGS_ROOT_DIR[:-1]
    RUNBOOKS_ROOT_DIR = os.path.abspath(args.runbooks)
    if RUNBOOKS_ROOT_DIR.endswith("/"):
        RUNBOOKS_ROOT_DIR = RUNBOOKS_ROOT_DIR[:-1]

    OCP_DOCS_VERSION = args.ocp_version
    HERMETIC_BUILD = args.hermetic_build

    os.environ["HF_HOME"] = args.model_dir
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    Settings.chunk_size = args.chunk
    Settings.chunk_overlap = args.overlap
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.model_dir)
    Settings.llm = resolve_llm(None)

    embedding_dimension = len(Settings.embed_model.get_text_embedding("random text"))
    faiss_index = faiss.IndexFlatIP(embedding_dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load documents
    documents = SimpleDirectoryReader(
        args.folder, recursive=True, file_metadata=ocp_file_metadata_func
    ).load_data()

    # Section-aware chunking with hierarchy preservation
    print(f"\nLoaded {len(documents)} documents. Chunking with section awareness...")
    good_nodes = []
    for doc in documents:
        text = doc.get_content()
        metadata = doc.metadata or {}
        chunks = chunk_document(text, max_tokens=args.chunk, min_words=50)
        nodes = chunks_to_text_nodes(chunks, metadata)
        good_nodes.extend(nodes)

    print(f"Created {len(good_nodes)} chunks from {len(documents)} documents")

    runbook_documents = SimpleDirectoryReader(
        args.runbooks,
        recursive=True,
        required_exts=[".md"],
        file_extractor={".md": FlatReader()},
        file_metadata=runbook_file_metadata_func,
    ).load_data()
    runbook_nodes = Settings.text_splitter.get_nodes_from_documents(runbook_documents)

    good_nodes.extend(runbook_nodes)

    # Chunk statistics
    word_counts = sorted(len(n.text.split()) for n in good_nodes)
    total = len(word_counts)
    buckets = {"<20": 0, "20-49": 0, "50-99": 0, "100-199": 0, "200-299": 0, "300+": 0}
    for wc in word_counts:
        if wc < 20: buckets["<20"] += 1
        elif wc < 50: buckets["20-49"] += 1
        elif wc < 100: buckets["50-99"] += 1
        elif wc < 200: buckets["100-199"] += 1
        elif wc < 300: buckets["200-299"] += 1
        else: buckets["300+"] += 1

    print(f"\nChunk statistics ({total} chunks):")
    print(f"  Min: {word_counts[0]} words, Max: {word_counts[-1]} words, "
          f"Avg: {sum(word_counts)//total} words, Median: {word_counts[total//2]} words")
    print(f"  Size distribution:")
    for label, count in buckets.items():
        pct = count * 100 // total
        bar = "#" * (pct // 2)
        print(f"    {label:>10}: {count:5d} ({pct:2d}%) {bar}")

    # Heading level vs chunk size analysis
    level_stats = {}  # level -> list of word counts
    no_header = []
    for n in good_nodes:
        text = n.text.strip()
        wc = len(text.split())
        # Check first line for heading level
        first_line = text.split("\n")[0]
        header_match = re.match(r'^(#+)\s', first_line)
        if header_match:
            level = len(header_match.group(1))
            level_stats.setdefault(level, []).append(wc)
        else:
            no_header.append(wc)

    print(f"\n  Heading level vs size (chunks starting with # heading):")
    for level in sorted(level_stats.keys()):
        wcs = sorted(level_stats[level])
        total_l = len(wcs)
        small = sum(1 for w in wcs if w < 50)
        print(f"    H{level} ({total_l:4d} chunks): "
              f"avg={sum(wcs)//total_l:3d}w, median={wcs[total_l//2]:3d}w, "
              f"<50w={small} ({small*100//total_l}%)")
    if no_header:
        small = sum(1 for w in no_header if w < 50)
        print(f"    No heading ({len(no_header):4d} chunks): "
              f"avg={sum(no_header)//len(no_header):3d}w, median={sorted(no_header)[len(no_header)//2]:3d}w, "
              f"<50w={small} ({small*100//len(no_header)}%)")

    batch_size = 2048
    total_batches = (len(good_nodes) + batch_size - 1) // batch_size
    print(f"\nEmbedding {len(good_nodes)} chunks in {total_batches} batches of {batch_size}...")

    # Embed in batches with outer progress tracking
    index = None
    for batch_num in range(0, len(good_nodes), batch_size):
        batch = good_nodes[batch_num:batch_num + batch_size]
        batch_idx = batch_num // batch_size + 1
        print(f"\n[Batch {batch_idx}/{total_batches}]")
        if index is None:
            index = VectorStoreIndex(
                batch,
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            index.insert_nodes(batch, show_progress=True)
    index.set_index_id(args.index)
    index.storage_context.persist(persist_dir=PERSIST_FOLDER)

    metadata: dict = {}
    metadata["execution-time"] = time.time() - start_time
    metadata["llm"] = "None"
    metadata["embedding-model"] = args.model_name
    metadata["index-id"] = args.index
    metadata["vector-db"] = "faiss.IndexFlatIP"
    metadata["embedding-dimension"] = embedding_dimension
    metadata["chunk"] = args.chunk
    metadata["overlap"] = args.overlap
    metadata["total-embedded-files"] = len(documents)

    with open(os.path.join(PERSIST_FOLDER, "metadata.json"), "w", encoding="utf-8") as file:
        file.write(json.dumps(metadata))

    if UNREACHABLE_DOCS > 0:
        print(
            "WARNING:\n"
            "There were documents with %s unreachable URLs, "
            "grep the log for UNREACHABLE.\n"
            "Please update the plain text." % UNREACHABLE_DOCS
        )
