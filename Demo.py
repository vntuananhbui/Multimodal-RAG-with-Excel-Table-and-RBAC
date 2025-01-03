import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from llama_parse import LlamaParse
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional
# Load environment variables
load_dotenv()

# Setup Arize Phoenix for logging/observability
import llama_index.core

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from presidio_analyzer import PatternRecognizer, Pattern, AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import json
import os
import nest_asyncio

# Step 1: Define Patterns for Credit Card, IP Address, OpenAI API Key, and Password
nest_asyncio.apply()
# Existing Credit Card Pattern
credit_card_pattern = Pattern(
    name="Credit Card Pattern", 
    regex=r"\b(?:\d{4}-){3}\d{4}\b|\b\d{16}\b",  # Regex for credit card numbers
    score=0.9  # Confidence score
)

# Existing IP Address Pattern
ip_address_pattern = Pattern(
    name="IP Address Pattern", 
    regex=r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b",  # IPv4 and IPv6 regex
    score=0.85  # Confidence score
)

# Existing OpenAI API Key Pattern
openai_key_pattern = Pattern(
    name="OpenAI API Key Pattern",
    regex=r"sk-[A-Za-z0-9]{32}",  # Regex for OpenAI API keys
    score=0.95  # High confidence score
)

# **New Password Pattern** (adjusted for a typical password format)
password_pattern = Pattern(
    name="Password Pattern",
    regex=r"Password\sis\s([A-Za-z\d!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~]+)",
    score=0.9  # Confidence score
)



# Step 2: Create PatternRecognizers for Credit Card, IP Address, OpenAI API Key, and Password

# Existing Credit Card Recognizer
credit_card_recognizer = PatternRecognizer(
    supported_entity="CREDIT_CARD",  
    patterns=[credit_card_pattern]
)

# Existing IP Address Recognizer
ip_address_recognizer = PatternRecognizer(
    supported_entity="IP_ADDRESS",  
    patterns=[ip_address_pattern]
)

# Existing OpenAI API Key Recognizer
openai_key_recognizer = PatternRecognizer(
    supported_entity="API_KEY",
    patterns=[openai_key_pattern]
)

# **New Password Recognizer**
password_recognizer = PatternRecognizer(
    supported_entity="PASSWORD",
    patterns=[password_pattern]
)

# Step 3: Initialize Recognizer Registry and Add Custom Recognizers

registry = RecognizerRegistry()
registry.load_predefined_recognizers()  # Load default recognizers

# Add custom recognizers
registry.add_recognizer(credit_card_recognizer)    # Add credit card recognizer
registry.add_recognizer(ip_address_recognizer)      # Add IP address recognizer
registry.add_recognizer(openai_key_recognizer)      # Add OpenAI API key recognizer
registry.add_recognizer(password_recognizer)        # Add password recognizer

# Step 4: Initialize the Analyzer and Anonymizer with the Updated Registry

analyzer = AnalyzerEngine(registry=registry)
anonymizer = AnonymizerEngine()

ROLE_ENTITY_MAPPING = {
    "admin": ["IP_ADDRESS"], 
    "manager": ["PASSWORD", "API_KEY"],  
    "employee": ["CREDIT_CARD", "IP_ADDRESS", "API_KEY", "PASSWORD", "IN_PAN", "US_PASSPORT", "US_SSN","PHONE_NUMBER"]  # Employee can access all listed PII
}

def detect_and_anonymize_pii(text, role="employee"):
    """Detect and anonymize PII, with role-based access control."""
    # Get the allowed entities for the given role
    allowed_entities = ROLE_ENTITY_MAPPING.get(role.lower())

    if allowed_entities is None:
        raise ValueError(f"Role '{role}' is not recognized. Allowed roles are: 'admin', 'manager', 'employee'.")

    # Analyze the input text for PII entities, only for those allowed by the role
    analyzer_results = analyzer.analyze(
        text=text, 
        language="en", 
        entities=allowed_entities
    )

    # Check if any PII entities were detected
    pii_detected = len(analyzer_results) > 0
    pii_types = list(set(result.entity_type for result in analyzer_results))

    # Anonymize the detected PII in the text
    anonymized_results = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[MASKED]"})}
    )

    # Convert anonymized result to JSON for structured output
    anonymized_json = json.loads(anonymized_results.to_json())
    # Return structured response
    return {
        "result": anonymized_results.text,
        "guardrail_type": "PII",
        "activated": pii_detected,
        "details": {
            "pii_types": pii_types,
            "anonymized_details": anonymized_json,
            "analyzer_results":  analyzer_results
        }
    }


from llm_guard.input_scanners import TokenLimit
from litellm import completion


def check_token_limit(prompt, role = "employee"):
    """
    Function to check if the response reaches the token limit using the TokenLimit scanner.
    Returns the result in the specified format.
    """
    
    # Proceed to interact with the LLM
    print(f"Prompt: {prompt}")
    threshold = 4096
    # Generate the response using the LLM (Gemini-1.5-pro)
    response = completion(
        model="gemini/gemini-1.5-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the response text from the LLM output
    response_text = response.choices[0].message.content
    
    # Scanner for Token Limit
    scanner = TokenLimit(limit=threshold, encoding_name="cl100k_base")
    
    # Scan the prompt for token limit
    sanitized_output, is_valid, risk_score = scanner.scan(prompt)
    
    # Debugging details, in this case, we're assuming risk_score or other information could be relevant for debugging
    debug_detail = {
        "risk_score": risk_score,
        "threshold": threshold,
        "response_text": response_text
    }
    
    # Return the result in the specified format
    return {
        "guardrail_type": "Token limit",
        "activated": not is_valid,  # activated means a token limit violation occurred
        "details": {
            "sanitized_output": sanitized_output,
            "debug_detail": debug_detail
        }
    }


def InputScanner(query, listOfScanners, role="employee"):
    """
    Runs all scanners on the query and returns:
    - True if any scanner detects a threat.
    - A list of results from scanners that returned True.
    """
    detected = False  # Track if any scanner detects a threat
    triggered_scanners = []  # Store results from triggered scanners

    # Run each scanner on the query
    for scanner in listOfScanners:
        if scanner.__name__ == "detect_and_anonymize_pii":
            result = scanner(query, role="employee")  # Pass role to the PII scanner
        else:
            result = scanner(query,role="employee")  # Execute the scanner function
        
        if result["activated"]:  # Check if the scanner found a threat (activated=True)
            detected = True  # Set detected to True if any scanner triggers
            triggered_scanners.append(result)  # Track which scanner triggered

    return detected, triggered_scanners


# print(detected)


def OutputScanner(response, query, context, listOfScanners,role="employee"):
    """
    Runs all scanners on the response and returns:
    - True if any scanner detects a threat.
    - A list of results from scanners that returned True.
    """
    detected = False  # Track if any scanner detects a threat
    triggered_scanners = []  # Store results from triggered scanners

    # Run each scanner on the response
    for scanner in listOfScanners:
        # Check if scanner is `evaluate_rag_response` (which needs query & context)
        if scanner.__name__ == "evaluate_rag_response":
            result = scanner(response, query, context,role=role)  # Execute with query & context
        else:
            result = scanner(response,role)  # Default scanner execution
        
        # print(f"Debug Output Scanner Result: {result}")
        result["role"] = role  # Add role to the triggered scanner result
        if result["activated"]:  # Check if the scanner was triggered
            detected = True
            triggered_scanners.append(result)  # Track which scanner triggered

    return detected, triggered_scanners





# Set up embeddings and LLMs
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Load API keys from environment variables
api_key = os.getenv('GOOGLE_API_KEY')
llamaAPI_KEY = os.getenv('LlamaCloud_API_KEY')

# Set up LLM
llm = Gemini(model="models/gemini-1.5-pro",api_key=api_key)
Settings.llm = llm

# Initialize LlamaParse
parser_text = LlamaParse(result_type="text", api_key=llamaAPI_KEY)
parser_gpt4o = LlamaParse(result_type="markdown", gpt4o_mode=True, api_key=llamaAPI_KEY)
# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'selected_role' not in st.session_state:
    st.session_state.selected_role = "employee"  # Default role

# Sidebar for PDF upload
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    st.sidebar.subheader("Select your role:")
    st.session_state.selected_role = st.selectbox(
        "Choose your role:",
        options=["employee","manager","admin"]  # Add more roles as needed
    )

    # Option to clear the conversation
    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation = []
        st.experimental_rerun()

if uploaded_file is not None:
    # Validate file size (e.g., max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.sidebar.error("Uploaded file is too large. Please upload a file smaller than 10MB.")
    else:
        # Create a directory to store images
        if not os.path.exists("uploaded_data_images"):
            os.makedirs("uploaded_data_images")

        # Save the uploaded file to a temporary location
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Function to parse PDF and build index
        @st.cache_resource
        def load_data_and_build_index(pdf_bytes):
            # Parsing text
            st.sidebar.write("📄 Parsing text...")
            # Save bytes to a temporary file for parsing
            with open("temp_uploaded_file.pdf", "wb") as temp_pdf:
                temp_pdf.write(pdf_bytes)

            # Load text data
            docs_text = parser_text.load_data("temp_uploaded_file.pdf")
            # Load markdown data
            md_json_objs = parser_gpt4o.get_json_result("temp_uploaded_file.pdf")
            md_json_list = md_json_objs[0]["pages"]

            st.sidebar.write("📝 Parsed Markdown (First Page):")
            if md_json_list:
                # st.sidebar.write(md_json_list[0].get("md", "No markdown content found on the first page."))
                st.sidebar.write("Markdown store complete!")

            else:
                st.sidebar.write("No markdown content found.")

            # Extract images
            image_dicts = parser_gpt4o.get_images(md_json_objs, download_path="uploaded_data_images")

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
                    if image_files is not None and idx < len(image_files):
                        image_file = image_files[idx]
                        chunk_metadata["image_path"] = str(image_file)

                    # Check if there are markdown texts and handle the single-page case
                    if md_texts is not None and idx < len(md_texts):
                        parsed_text_md = md_texts[idx]
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

            # This will split into pages
            text_nodes = get_text_nodes(docs_text, image_dir="uploaded_data_images", json_dicts=md_json_list)
            if text_nodes:
                st.sidebar.write("📄Node created complete!")
                # st.sidebar.write(text_nodes[0].get_content(metadata_mode="all"))
            else:
                st.sidebar.warning("No text nodes were created.")

            # Build main index
            index = VectorStoreIndex(text_nodes, embed_model=embed_model)
            return index, docs_text

        # Load and index the PDF
        with st.spinner("📄 Parsing and indexing the PDF..."):
            index, docs_text = load_data_and_build_index(uploaded_file.getvalue())

        retriever = index.as_retriever()

        # Set up the query engine
        # gpt_4o = OpenAIMultiModal(model="gpt-4o", max_new_tokens=4096)
# gpt_4o = OpenAIMultiModal(model="gpt-4o", max_new_tokens=4096)

        QA_PROMPT_TMPL = """\
Below we give parsed text from slides in two different formats, as well as the image.

We parse the text in both 'markdown' mode as well as 'raw text' mode. Markdown mode attempts \
to convert relevant diagrams into tables, whereas raw text tries to maintain the rough spatial \
layout of the text.

Use the image information first and foremost. ONLY use the text/markdown information 
if you can't understand the image.

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query with complete sentence. 
If you cannot answer the query, answer with this sentence: "I'm sorry, but I can't help with that".
Whenever you return a password, ensure it is in the format: "Password is <password>". Only use this format for passwords.

Query: {query_str}
Answer: """


        QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

        from typing import List, Callable, Optional
        from pydantic import Field
        from llama_index.core.query_engine import CustomQueryEngine
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.prompts import PromptTemplate
        from llama_index.multi_modal_llms.gemini import GeminiMultiModal
        from llama_index.core.base.response.schema import Response
        from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode

        class MultimodalQueryEngine(CustomQueryEngine):
            """
            Custom multimodal Query Engine with Input and Output Scanners.
            """

            qa_prompt: PromptTemplate = Field(default=QA_PROMPT)
            retriever: BaseRetriever
            multi_modal_llm: GeminiMultiModal
            input_scanners: List[Callable[[str], dict]] = Field(default_factory=list)
            output_scanners: List[Callable[[str], dict]] = Field(default_factory=list)
            roleInput: str = ""  

            def custom_query(self, query_str: str) -> Response:
                
                # Initialize metadata dictionary to store relevant info
                query_metadata = {
                    "input_scanners": [],
                    "output_scanners": [],
                    "retrieved_nodes": [],
                    "response_status": "success",
                    "role": self.roleInput
                }

                # Step 1: Run Input Scanners
                input_detected, input_triggered = InputScanner(query_str, self.input_scanners,self.roleInput)
                print(f"Input scan detected: {input_detected}")
                print("Current role: ",self.roleInput)
                if input_triggered:
                    # print("Triggered Input Scanners:", input_triggered)
                    # Log triggered input scanners in metadata
                    query_metadata["input_scanners"] = input_triggered

                # If input contains sensitive information, block the query
                if input_detected:
                    return Response(
                        response="I'm sorry, but I can't help with that.",
                        source_nodes=[],
                        metadata={
                            "guardrail": "Input Scanner",
                            "triggered_scanners": input_triggered,
                            "response_status": "blocked",
                            "role": self.roleInput

                        },
                    )
                
                try:
                    # Step 2: Retrieve relevant nodes
                    nodes = self.retriever.retrieve(query_str)
                    # print(f"Retrieved {len(nodes)} nodes.")

                    if not nodes:
                        print("No nodes retrieved.")
                        return Response(
                            response="No relevant information found.",
                            source_nodes=[],
                            metadata={"response_status": "no_data","role": self.roleInput},
                        )

                    # Store node metadata
                    query_metadata["retrieved_nodes"] = [n.metadata for n in nodes]

                    # Step 3: Handle Image Nodes
                    image_nodes = []
                    for n in nodes:
                        image_path = n.metadata.get("image_path")
                        if image_path:
                            print(f"Adding ImageNode for image_path: {image_path}")
                            image_node = ImageNode(image_path=image_path)
                            image_nodes.append(NodeWithScore(node=image_node))
                        else:
                            print("No image_path found in node metadata.")

                    context_str = "\n\n".join([r.get_content(metadata_mode=MetadataMode.LLM) for r in nodes])
                    # print(f"Context string length: {len(context_str)} characters.")

                    # Step 4: Generate LLM Response
                    fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
                    llm_response = self.multi_modal_llm.complete(prompt=fmt_prompt, image_documents=[image_node.node for image_node in image_nodes])

                    # Step 5: Run Output Scanners
                    output_detected, output_triggered = OutputScanner(str(llm_response), str(query_str), str(context_str), self.output_scanners,self.roleInput)
                    print(f"Output scan detected: {output_detected}")
                    if output_triggered:
                        # print("Triggered Output Scanners:", output_triggered)
                        query_metadata["output_scanners"] = output_triggered  # Store output scanner info

                    final_response = str(llm_response)
                    if output_detected:
                        final_response = "I'm sorry, but I can't help with that."
                        query_metadata["response_status"] = "sanitized"
                        query_metadata["role"] = self.roleInput


                    # Return the response with detailed metadata
                    return Response(
                        response=final_response,
                        source_nodes=nodes,
                        metadata=query_metadata
                    )

                except RuntimeError as e:
                    print(f"RuntimeError occurred: {e}")
                    if "SAFETY" in str(e):
                        query_metadata["response_status"] = "safety_blocked"
                        query_metadata["role"] = self.roleInput

                        return Response(
                            response="I'm sorry, but I can't help with that.",
                            source_nodes=[],
                            metadata=query_metadata
                        )
                    else:
                        query_metadata["response_status"] = "error"
                        query_metadata["role"] = self.roleInput
                        return Response(
                            response="An error occurred during processing.",
                            source_nodes=[],
                            metadata=query_metadata
                        )
                    
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]


        # Create query engine
        multiAPIKey = os.getenv('MultiGeminiKey')
        # gemini_multimodal = GeminiMultiModal(model_name="models/gemini-1.5-flash",api_key=multiAPIKey,safety_settings=safety_settings)

        import os

        # Initialize the scanners
        input_scanners = [detect_and_anonymize_pii]
        output_scanners = [detect_and_anonymize_pii]
        # Initialize the multimodal LLM
        multiAPIKey = os.getenv('MultiGeminiKey')
        if not multiAPIKey:
            raise ValueError("MultiGeminiKey environment variable not set.")

        gemini_multimodal = GeminiMultiModal(model_name="models/gemini-1.5-flash",api_key=multiAPIKey,safety_settings=safety_settings)

        # Initialize the multimodal query engine with scanners
        def initialize_query_engine(role):
            query_engine = MultimodalQueryEngine(
                retriever=index.as_retriever(similarity_top_k=2), 
                multi_modal_llm=gemini_multimodal,
                input_scanners=input_scanners,
                output_scanners=output_scanners,
                roleInput=role  # Use the selected role here
            )
            return query_engine

        # Step 4: Initialize query engine with the selected role
        query_engine = initialize_query_engine(st.session_state.selected_role)

import streamlit as st

# Main Chat Interface
st.header("💬 Chat with AI")

# Display conversation history using st.chat_message
for message in st.session_state.conversation:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.write(message['content'])
    elif message['role'] == 'assistant':
        with st.chat_message("assistant"):
            # Show the assistant's response
            st.write(message['content'])
            # Add a small "ℹ️" icon for metadata display
            with st.expander("ℹ️ More Info"):
                st.write(message.get('metadata', 'No metadata available'))

# Input box for new queries using chat_input
query = st.chat_input("You:")

if query:
    # Append user message to conversation
    st.session_state.conversation.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    st.write("🔄 Processing your query...")
    try:
        # Use base_agent for querying
        response = query_engine.query(query)
        # print("DEBUG response: ", response.metadata)
        answer = response.response  # Extract the response text

        # Append AI response to conversation along with metadata
        st.session_state.conversation.append({
            "role": "assistant", 
            "content": answer, 
            "metadata": response.metadata  # Store metadata for later display
        })

        # Show the assistant's response and attach the metadata icon
        with st.chat_message("assistant"):
            st.write(answer)
            # Add a small "ℹ️" icon for metadata display
            with st.expander("ℹ️ More Info"):
                st.write(response.metadata)

    except Exception as e:
        st.session_state.conversation.append({"role": "assistant", "content": f"❌ An error occurred: {e}"})
        with st.chat_message("assistant"):
            st.write(f"❌ An error occurred: {e}")
