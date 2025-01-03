from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
# from llama_parse import LlamaParse
import nest_asyncio
import os
load_dotenv()
nest_asyncio.apply()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.embed_model = embed_model


api_key = os.getenv('GOOGLE_API_KEY')
llamaAPI_KEY = os.getenv('LlamaCloud_API_KEY')
llm = Gemini(model="models/gemini-1.5-flash",api_key=api_key)
Settings.llm = llm
from llama_parse import LlamaParse


parser_text = LlamaParse(result_type="text",api_key=llamaAPI_KEY)
parser_gpt4o = LlamaParse(result_type="markdown", gpt4o_mode=True,api_key=llamaAPI_KEY)
print(f"Parsing text...")
docs_text = parser_text.load_data("/Users/macintosh/TA-DOCUMENT/StudyZone/ComputerScience/Project_RAG/uploaded_file.pdf")
md_json_objs = parser_gpt4o.get_json_result("/Users/macintosh/TA-DOCUMENT/StudyZone/ComputerScience/Project_RAG/uploaded_file.pdf")
md_json_list = md_json_objs[0]["pages"]
# print(md_json_list[0]["md"])
image_dicts = parser_gpt4o.get_images(md_json_objs, download_path="data_images")
from llama_index.core.schema import TextNode
from typing import Optional
# get pages loaded through llamaparse
import re


def get_page_number(file_name):
    match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0


def _get_sorted_image_files(image_dir):
    """Get image files sorted by page."""
    raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
    sorted_files = sorted(raw_files, key=get_page_number)
    return sorted_files
from copy import deepcopy
from pathlib import Path


from copy import deepcopy
from pathlib import Path

# Assuming TextNode class is defined somewhere else in your code
# Attach image metadata to the text nodes
def get_text_nodes(docs, image_dir=None, json_dicts=None):
    """Split docs into nodes, by separator."""
    nodes = []

    # Get image files (if provided)
    image_files = _get_sorted_image_files(image_dir) if image_dir is not None else None

    # Get markdown texts (if provided)
    md_texts = [d["md"] for d in json_dicts] if json_dicts is not None else None

    # Split docs into chunks by separator
    doc_chunks = [c for d in docs for c in d.text.split("---")]

    # Handle both single-page and multi-page cases
    for idx, doc_chunk in enumerate(doc_chunks):
        chunk_metadata = {"page_num": idx + 1}
        
        # Check if there are image files and handle the single-page case
        if image_files is not None:
            # Use the first image file if there's only one
            image_file = image_files[idx] if idx < len(image_files) else image_files[0]
            chunk_metadata["image_path"] = str(image_file)
        
        # Check if there are markdown texts and handle the single-page case
        if md_texts is not None:
            # Use the first markdown text if there's only one
            parsed_text_md = md_texts[idx] if idx < len(md_texts) else md_texts[0]
            chunk_metadata["parsed_text_markdown"] = parsed_text_md

        # Add the chunk text as metadata
        chunk_metadata["parsed_text"] = doc_chunk

        # Create the TextNode with the parsed text and metadata
        node = TextNode(
            text="",
            metadata=chunk_metadata,
        )
        nodes.append(node)

    return nodes

# this will split into pages
text_nodes = get_text_nodes(docs_text, image_dir="data_images", json_dicts=md_json_list)
# text_nodes = get_text_nodes(docs_text, image_dir=None, json_dicts=md_json_list)
print(str(text_nodes[0].get_content(metadata_mode="parsed_text")))