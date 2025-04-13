import os
import json
import re
import zipfile
import tempfile
from io import BytesIO
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import docx2txt
from PIL import Image
import pytesseract

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# ---------- Helper Functions ----------
def clean_extraction_result(result: str) -> str:
    """
    Removes markdown code fences (like ```json ... ```) from the LLM output.
    """
    result = result.strip()
    if result.startswith("```"):
        # Split lines and remove the fence lines
        lines = result.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        result = "\n".join(lines).strip()
    return result

def extract_metadata_from_path(folder_path: str) -> Dict[str, str]:
    """
    Extract PO number, SF code, and document type from folder path.
    Expected format: .../PO_1498093_SF_800056/01_PACKING_LIST/...
    Also handles formats like PO_1505749_SF_ and PO_1511159_SF_S3161
    """
    metadata = {}
    
    # Extract PO number and SF code from the parent folder
    parent_folder = os.path.basename(os.path.dirname(folder_path))
    po_match = re.search(r'PO_(\d+)', parent_folder, re.IGNORECASE)
    sf_match = re.search(r'SF_([^/\\]*)', parent_folder, re.IGNORECASE)
    
    if po_match:
        metadata['po_number'] = po_match.group(1)
    if sf_match:
        sf_code = sf_match.group(1).strip()
        if sf_code:  # Only add if not empty
            metadata['sf_code'] = sf_code
        else:
            metadata['sf_code'] = "unknown"  # Default value for empty SF code
    
    # Extract document type from the current folder
    folder_name = os.path.basename(folder_path)
    doc_type_match = re.search(r'\d+_(.+)', folder_name, re.IGNORECASE)
    
    if doc_type_match:
        doc_type = doc_type_match.group(1).lower().replace('_', ' ')
        metadata['doc_type'] = doc_type
        
    return metadata

# ---------- Unified Document Loader ----------
def load_document(filepath):
    """
    Loads a document file and returns a list of document objects.
    Each object is expected to have page_content and metadata.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    # Base metadata with file path information
    metadata = {
        "source_file": os.path.basename(filepath),
        "path": filepath,
    }
    
    # Try to extract additional metadata from the path if applicable
    folder_path = os.path.dirname(filepath)
    path_metadata = extract_metadata_from_path(folder_path)
    metadata.update(path_metadata)
    
    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
        docs = loader.load()  # Returns list of document objects
        # Add our metadata to each document
        for doc in docs:
            doc.metadata.update(metadata)
        return docs
    elif ext in [".docx", ".doc"]:
        text = docx2txt.process(filepath)
        return [Document(page_content=text, metadata=metadata)]
    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img)
        except Exception as e:
            text = ""
        return [Document(page_content=text, metadata=metadata)]
    elif ext in [".txt"]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [Document(page_content=text, metadata=metadata)]
    else:
        return []

def process_folder_structure(root_folder):
    """
    Process all documents within a hierarchical folder structure:
    - root_folder/
      - PO_1234_SF_5678/
        - 01_PACKING_LIST/
          - document.pdf
        - 02_INVOICE/
          - other_document.pdf
    
    Returns a list of Document objects with metadata extracted from the folder structure.
    """
    all_docs = []
    
    # Find all PO_*_SF_* folders with more flexible pattern matching
    po_folders = []
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        # More flexible regex that handles any format after SF_
        if os.path.isdir(item_path) and re.match(r'PO_\d+_SF_', item, re.IGNORECASE):
            po_folders.append(item_path)
    
    # Process each PO folder
    for po_folder in po_folders:
        # Find all document type subfolders (01_PACKING_LIST, etc.)
        for item in os.listdir(po_folder):
            subfolder_path = os.path.join(po_folder, item)
            if os.path.isdir(subfolder_path) and re.match(r'\d+_.+', item):
                # Process all documents in this subfolder
                for filename in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, filename)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(filename)[1].lower()
                        supported_extensions = [".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg"]
                        if ext in supported_extensions:
                            docs = load_document(file_path)
                            all_docs.extend(docs)
    
    return all_docs

def extract_zip_and_process(zip_file):
    """
    Extract a zip file and process its contents using the folder structure.
    Returns a list of Document objects.
    """
    all_docs = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save uploaded zip to temporary directory
        zip_path = os.path.join(tmpdirname, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
            
        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)
            
        # Process the extracted folder structure
        all_docs = process_folder_structure(tmpdirname)
            
    return all_docs

# ---------- LLM Extraction Setup ----------
extraction_template = """
You are an assistant that extracts shipping package information from the provided text.
A deliver can include one or more packages (or boxes). For each package, extract the following details:
- PO/TR No.
- SF No.
- WAYBILL N° (B/L/AWB)
- VENDOR
- GROSS WEIGHT KG
- Dims. (Length Cm.)
- Dims. (Width Cm.)
- Dims. (Height Cm.)

Note: Some fields (like PO/TR No., SF No., WAYBILL, and VENDOR) may be common for all packages within the deliver.
However, package-specific details like GROSS WEIGHT and dimensions might differ.
For each distinct package, output an object with these keys:
"PO_TR_No", "SF_No", "WAYBILL", "VENDOR", "GROSS_WEIGHT_KG", "Dims_Length_Cm", "Dims_Width_Cm", "Dims_Height_Cm".

Output the result as a JSON array.

Text:
{text}

JSON Array:
"""
prompt_template = PromptTemplate(template=extraction_template, input_variables=["text"])
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
extraction_chain = LLMChain(llm=llm, prompt=prompt_template)

# ---------- RAG Setup ----------
def setup_rag_with_documents(docs):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Setup conversation memory with return messages for better context preservation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    
    # Create a more detailed system prompt that includes PO/SF recognition
    system_template = """You are a helpful assistant specialized in answering questions about shipping documents.
    
    Each document has metadata that you MUST use to provide accurate answers:
    - po_number: The Purchase Order number that identifies a specific delivery/shipment
    - sf_code: The SF code that identifies a specific delivery/shipment
    - doc_type: The type of document (e.g., packing list, invoice)
    - source_file: The original filename
    
    IMPORTANT: Each unique combination of PO number and SF code represents a DISTINCT delivery.
    When users ask about "deliveries" or "delivers", they are referring to these PO/SF combinations.
    
    If asked about specific PO numbers or SF codes, ONLY provide information from documents with those matching metadata.
    If asked about all deliveries, summarize information from all available PO/SF combinations.
    
    Always maintain conversation context. When the user asks follow-up questions or refers to previous questions,
    use your memory of the conversation to provide contextually appropriate responses.
    
    Organize your answers by delivery (PO/SF) when appropriate, and be explicit about which delivery you're discussing.
    
    When uncertain about a delivery's details, check your context for that specific PO/SF combination.
    
    Make sure to ALWAYS keep track of the conversation history and reference previous questions when asked.
    """
    
    # Create chat model
    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", 
        temperature=0
    )
    
    # Create the RAG chain with enhanced prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        memory=memory,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={
            "document_variable_name": "context",
        }
    )
    
    return qa_chain

# ---------- Enhanced RAG Setup with Full Document Context ----------
def setup_enhanced_rag_with_documents(docs, use_all_context=False):
    """
    Setup an enhanced RAG system that either uses standard chunking or preserves full document context
    
    Args:
        docs: List of document objects
        use_all_context: If True, uses a special retrieval method that provides more complete document context
    """
    # Store all original documents in session state for reference
    if "all_documents" not in st.session_state:
        st.session_state.all_documents = []
    
    # Add new documents to the global collection
    st.session_state.all_documents.extend(docs)
    
    # Process documents based on context strategy
    if use_all_context:
        # For full context retrieval, use larger chunks with more overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    else:
        # Standard chunking for regular retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    chunks = text_splitter.split_documents(docs)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Setup conversation memory with return messages for better context preservation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    
    # Create a more detailed system prompt that includes PO/SF recognition
    system_template = """You are a helpful assistant specialized in answering questions about documents.
    
    Each document has metadata that you should use to provide accurate answers:
    - po_number: The Purchase Order number that identifies a specific delivery/shipment
    - sf_code: The SF code that identifies a specific delivery/shipment
    - doc_type: The type of document (e.g., packing list, invoice)
    - source_file: The original filename
    
    IMPORTANT: Each unique combination of PO number and SF code represents a DISTINCT delivery.
    When users ask about "deliveries" or "delivers", they are referring to these PO/SF combinations.
    
    If asked about specific PO numbers or SF codes, ONLY provide information from documents with those matching metadata.
    If asked about all deliveries, summarize information from ALL available PO/SF combinations.
    
    Always maintain conversation context. When the user asks follow-up questions or refers to previous questions,
    use your memory of the conversation to provide contextually appropriate responses.
    
    When asked about "all documents" or "everything", make sure to consider ALL documents in your answer.
    
    Organize your answers by delivery (PO/SF) when appropriate, and be explicit about which delivery you're discussing.
    """
    
    # Create chat model
    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", 
        temperature=0
    )
    
    # Create the RAG chain with enhanced prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10 if use_all_context else 6}),
        memory=memory,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={
            "document_variable_name": "context",
        }
    )
    
    return qa_chain

# ---------- Streamlit Interface ----------
st.set_page_config(page_title="Document Assistant", layout="wide")
st.title("Document Assistant")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "general_chat_history" not in st.session_state:
    st.session_state.general_chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "general_qa_chain" not in st.session_state:
    st.session_state.general_qa_chain = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
if "general_user_question" not in st.session_state:
    st.session_state.general_user_question = ""
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "general_user_input" not in st.session_state:
    st.session_state.general_user_input = ""
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Document Chat", "Delivers Processing", "General Document Assistant"])

# ---------- Tab 1: Document Chat ----------
with tab1:
    st.header("Chat with your Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Upload Documents")
        
        # Add option to select document source
        upload_option = st.radio(
            "Document source",
            ["Individual Files", "Folder Structure (ZIP)", "Local Folder"]
        )
        
        if upload_option == "Individual Files":
            # File uploader for individual documents
            uploaded_files = st.file_uploader(
                "Upload documents to chat with", 
                type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg"],
                accept_multiple_files=True
            )
            
            # Process uploaded files
            if uploaded_files and not st.session_state.documents_loaded:
                with st.spinner("Processing your documents..."):
                    temp_dir = tempfile.mkdtemp()
                    docs = []
                    
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        docs.extend(load_document(file_path))
                    
                    if docs:
                        st.session_state.qa_chain = setup_rag_with_documents(docs)
                        st.session_state.documents_loaded = True
                        st.success(f"{len(docs)} document(s) processed successfully!")
                    else:
                        st.error("No content could be extracted from the uploaded documents.")
        
        elif upload_option == "Folder Structure (ZIP)":
            # Add uploader for ZIP files containing the folder structure
            uploaded_zip = st.file_uploader(
                "Upload ZIP with folder structure", 
                type=["zip"]
            )
            
            if uploaded_zip and not st.session_state.documents_loaded:
                with st.spinner("Processing your documents from ZIP..."):
                    try:
                        docs = extract_zip_and_process(uploaded_zip)
                        
                        if docs:
                            st.session_state.qa_chain = setup_rag_with_documents(docs)
                            st.session_state.documents_loaded = True
                            
                            # Display metadata summary
                            metadata_summary = {}
                            for doc in docs:
                                po_number = doc.metadata.get('po_number', 'Unknown')
                                sf_code = doc.metadata.get('sf_code', 'Unknown')
                                doc_type = doc.metadata.get('doc_type', 'Unknown')
                                key = f"PO: {po_number}, SF: {sf_code}, Type: {doc_type}"
                                metadata_summary[key] = metadata_summary.get(key, 0) + 1
                            
                            st.success(f"{len(docs)} document(s) processed successfully!")
                            st.write("Document summary:")
                            for key, count in metadata_summary.items():
                                st.write(f"- {key}: {count} page(s)")
                        else:
                            st.error("No content could be extracted from the ZIP file.")
                    except Exception as e:
                        st.error(f"Error processing ZIP file: {str(e)}")
        
        elif upload_option == "Local Folder":
            # Input for local folder path
            folder_path = st.text_input("Enter path to documents folder")
            process_button = st.button("Process Folder")
            
            if folder_path and process_button and not st.session_state.documents_loaded:
                if os.path.isdir(folder_path):
                    with st.spinner("Processing documents from folder..."):
                        try:
                            docs = process_folder_structure(folder_path)
                            
                            if docs:
                                st.session_state.qa_chain = setup_rag_with_documents(docs)
                                st.session_state.documents_loaded = True
                                
                                # Display metadata summary
                                metadata_summary = {}
                                for doc in docs:
                                    po_number = doc.metadata.get('po_number', 'Unknown')
                                    sf_code = doc.metadata.get('sf_code', 'Unknown')
                                    doc_type = doc.metadata.get('doc_type', 'Unknown')
                                    key = f"PO: {po_number}, SF: {sf_code}, Type: {doc_type}"
                                    metadata_summary[key] = metadata_summary.get(key, 0) + 1
                                
                                st.success(f"{len(docs)} document(s) processed successfully!")
                                st.write("Document summary:")
                                for key, count in metadata_summary.items():
                                    st.write(f"- {key}: {count} page(s)")
                            else:
                                st.error("No content could be extracted from the folder.")
                        except Exception as e:
                            st.error(f"Error processing folder: {str(e)}")
                else:
                    st.error("Invalid folder path. Please provide a valid directory path.")
        
        # Clear buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.session_state.documents_loaded:
                if st.button("Clear Documents"):
                    st.session_state.documents_loaded = False
                    st.session_state.qa_chain = None
                    st.rerun()
        
        with col_b:
            if st.session_state.chat_history:
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    if st.session_state.qa_chain and hasattr(st.session_state.qa_chain, "memory"):
                        st.session_state.qa_chain.memory.clear()
                    st.rerun()
    
    with col1:
        # Chat interface
        st.subheader("Chat")
        
        # Display available deliveries when documents are loaded
        if st.session_state.documents_loaded:
            # Collect unique PO/SF combinations for context
            all_docs = []
            for value in st.session_state.qa_chain.retriever.vectorstore.docstore._dict.values():
                all_docs.append(value)
                
            deliveries = {}
            for doc in all_docs:
                po_number = doc.metadata.get('po_number', 'Unknown')
                sf_code = doc.metadata.get('sf_code', 'Unknown')
                delivery_key = f"PO_{po_number}_SF_{sf_code}"
                
                if delivery_key not in deliveries:
                    deliveries[delivery_key] = {
                        'po_number': po_number,
                        'sf_code': sf_code,
                        'doc_types': set()
                    }
                
                doc_type = doc.metadata.get('doc_type', 'Unknown')
                deliveries[delivery_key]['doc_types'].add(doc_type)
            
            st.write("#### Available Deliveries")
            st.info(f"You can ask questions about {len(deliveries)} deliveries:")
            for key, details in deliveries.items():
                st.write(f"- **{key}**: Document types: {', '.join(details['doc_types'])}")
            
            st.write("---")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.markdown(f"**You:** {message}")
                else:
                    st.markdown(f"**Assistant:** {message}")
        
        # Function to clear input after submission
        def submit_question():
            if st.session_state.user_input:
                st.session_state.user_question = st.session_state.user_input
                st.session_state.user_input = ""  # Clear the input field
        
        # User input with text area and submit button
        st.text_area("Ask a question about your documents:", 
                     key="user_input", 
                     height=100,
                     value=st.session_state.user_input)
        
        if st.button("Submit", on_click=submit_question):
            pass  # The actual action happens in the on_click callback
        
        # Process the question if there is one in the session state
        if st.session_state.user_question:
            if not st.session_state.documents_loaded:
                st.warning("Please upload documents first.")
                st.session_state.user_question = ""  # Clear the question
            else:
                with st.spinner("Thinking..."):
                    try:
                        # Store the question first before processing
                        question = st.session_state.user_question
                        st.session_state.chat_history.append(question)
                        
                        # Use invoke with the question
                        response = st.session_state.qa_chain.invoke({
                            "question": question
                        })
                        
                        # Extract the answer
                        answer = response.get("answer", "I'm sorry, I couldn't generate a response.")
                        source_docs = response.get("source_documents", [])
                        
                        # Get source information to display
                        sources_info = []
                        seen_sources = set()
                        for doc in source_docs:
                            po = doc.metadata.get('po_number', 'Unknown')
                            sf = doc.metadata.get('sf_code', 'Unknown')
                            doc_type = doc.metadata.get('doc_type', 'Unknown')
                            source_key = f"PO_{po}_SF_{sf} ({doc_type})"
                            
                            if source_key not in seen_sources:
                                sources_info.append(source_key)
                                seen_sources.add(source_key)
                        
                        # Add sources to the answer if available
                        if sources_info:
                            sources_text = "\n\n*Sources: " + ", ".join(sources_info) + "*"
                            answer += sources_text
                        
                        # Add to chat history
                        st.session_state.chat_history.append(answer)
                        
                        # Clear the question to prevent reprocessing
                        st.session_state.user_question = ""
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        # Remove the question from history if there was an error
                        if st.session_state.chat_history and st.session_state.chat_history[-1] == question:
                            st.session_state.chat_history.pop()
                        st.session_state.user_question = ""

# ---------- Tab 2: Delivers Processing ----------
with tab2:
    st.header("Delivers Processing")
    st.write("Upload documents for information extraction. You can use the folder structure to automatically extract metadata.")
    
    # Option to load existing Excel file
    existing_excel = st.file_uploader("Load existing Excel file (optional)", type=["xlsx"])
    existing_df = None
    
    if existing_excel:
        existing_df = pd.read_excel(existing_excel)
        st.success(f"Loaded existing Excel with {len(existing_df)} entries")
        st.dataframe(existing_df)
    
    # Document source selection
    extraction_source = st.radio(
        "Document source for extraction",
        ["Individual Files", "Folder Structure (ZIP)", "Local Folder"],
        key="extraction_source"
    )
    
    # Create a session state for documents to persist across reruns
    if "documents_for_extraction" not in st.session_state:
        st.session_state.documents_for_extraction = []
    
    if extraction_source == "Individual Files":
        uploaded_files = st.file_uploader(
            "Upload documents for extraction", 
            type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="extraction_files"
        )
        
        if uploaded_files:
            with st.spinner("Processing files for extraction..."):
                temp_dir = tempfile.mkdtemp()
                docs = []
                
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    docs.extend(load_document(file_path))
                
                st.session_state.documents_for_extraction = docs
                st.success(f"Processed {len(docs)} files for extraction")
    
    elif extraction_source == "Folder Structure (ZIP)":
        uploaded_zip = st.file_uploader(
            "Upload ZIP with folder structure", 
            type=["zip"],
            key="extraction_zip"
        )
        
        if uploaded_zip:
            with st.spinner("Processing ZIP for extraction..."):
                try:
                    docs = extract_zip_and_process(uploaded_zip)
                    st.session_state.documents_for_extraction = docs
                    st.success(f"Processed {len(docs)} documents from ZIP")
                except Exception as e:
                    st.error(f"Error processing ZIP: {str(e)}")
                    st.session_state.documents_for_extraction = []
    
    elif extraction_source == "Local Folder":
        folder_path = st.text_input("Enter path to documents folder", key="extraction_folder")
        process_button = st.button("Process Folder for Extraction")
        
        if folder_path and process_button:
            if os.path.isdir(folder_path):
                with st.spinner("Processing folder for extraction..."):
                    try:
                        docs = process_folder_structure(folder_path)
                        st.session_state.documents_for_extraction = docs
                        st.success(f"Processed {len(docs)} documents from folder")
                    except Exception as e:
                        st.error(f"Error processing folder: {str(e)}")
                        st.session_state.documents_for_extraction = []
            else:
                st.error("Invalid folder path. Please provide a valid directory path.")
    
    # Display document count
    if st.session_state.documents_for_extraction:
        st.info(f"{len(st.session_state.documents_for_extraction)} documents ready for extraction")
    
    # Extract information button
    extract_button = st.button("Extract Package Information")
    if extract_button:
        if not st.session_state.documents_for_extraction:
            st.warning("Please upload or process documents first")
        else:
            with st.spinner("Extracting information from documents..."):
                all_packages = []
                metadata_groups = {}
                
                # Group documents by metadata for context-aware processing
                for doc in st.session_state.documents_for_extraction:
                    po_number = doc.metadata.get('po_number', 'unknown')
                    sf_code = doc.metadata.get('sf_code', 'unknown')
                    doc_type = doc.metadata.get('doc_type', 'unknown')
                    
                    key = f"{po_number}_{sf_code}_{doc_type}"
                    if key not in metadata_groups:
                        metadata_groups[key] = []
                    metadata_groups[key].append(doc)
                
                # Progress bar
                progress_bar = st.progress(0)
                progress_text = st.empty()
                st.write(f"Processing {len(metadata_groups)} document groups...")
                
                # Process each group
                for i, (key, docs) in enumerate(metadata_groups.items()):
                    # Update progress
                    progress = (i + 1) / len(metadata_groups)
                    progress_bar.progress(progress)
                    progress_text.text(f"Processing group {i+1}/{len(metadata_groups)}: {key}")
                    
                    # Get metadata from the first document in group (they should all have the same)
                    metadata = docs[0].metadata
                    
                    # Show document being processed
                    st.write(f"Processing document group: {key}")
                    
                    # Combine text from all docs in this group
                    combined_text = "\n".join([doc.page_content for doc in docs])
                    
                    # If text is too long, trim it
                    if len(combined_text) > 12000:
                        st.write(f"⚠️ Text is very long ({len(combined_text)} chars), trimming to first 12000 chars")
                        combined_text = combined_text[:12000]
                    
                    # Run extraction
                    try:
                        extraction_result = extraction_chain.run(text=combined_text)
                        extraction_result_clean = clean_extraction_result(extraction_result)
                        
                        # Debug: Show the raw extraction result
                        with st.expander(f"Raw extraction result for {key}"):
                            st.code(extraction_result_clean)
                        
                        packages = json.loads(extraction_result_clean)
                        if isinstance(packages, dict):
                            packages = [packages]
                        
                        # Add metadata to each package
                        for pkg in packages:
                            # Use extracted metadata from folder structure if available
                            if 'po_number' in metadata:
                                pkg["PO_TR_No"] = pkg.get("PO_TR_No", "") or metadata['po_number']
                            if 'sf_code' in metadata:
                                pkg["SF_No"] = pkg.get("SF_No", "") or metadata['sf_code']
                            
                            # Add source file and doc type but not the full path
                            pkg["SOURCE_FILE"] = metadata.get('source_file', '')
                            pkg["DOC_TYPE"] = metadata.get('doc_type', '')
                            
                            all_packages.append(pkg)
                            
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing JSON from extraction result for {key}: {str(e)}")
                        st.error(f"Raw output: {extraction_result_clean}")
                    except Exception as e:
                        st.error(f"Error during extraction for {key}: {str(e)}")
                
                # Clear progress
                progress_bar.empty()
                progress_text.empty()
                
                if all_packages:
                    st.success(f"Successfully extracted information for {len(all_packages)} packages")
                    df = pd.DataFrame(all_packages)
                    rename_mapping = {
                        "PO_TR_No": "PO/TR No.",
                        "SF_No": "SF No.",
                        "WAYBILL": "WAYBILL N° (B/L/AWB)",
                        "GROSS_WEIGHT_KG": "GROSS WEIGHT KG",
                        "Dims_Length_Cm": "Dims. (Length Cm.)",
                        "Dims_Width_Cm": "Dims. (Width Cm.)",
                        "Dims_Height_Cm": "Dims. (Height Cm.)",
                        "SOURCE_FILE": "Source File",
                        "DOC_TYPE": "Document Type"
                    }
                    df.rename(columns=rename_mapping, inplace=True)
                    
                    # Combine with existing data if it exists
                    if existing_df is not None:
                        # Check for duplicates based on multiple columns
                        duplicate_columns = ["PO/TR No.", "SF No."]
                        if all(col in existing_df.columns for col in duplicate_columns):
                            # Create a mask for rows that aren't in the existing data
                            mask = ~df.set_index(duplicate_columns).index.isin(
                                existing_df.set_index(duplicate_columns).index
                            )
                            new_rows = df[mask].copy()
                            combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
                            st.success(f"Added {len(new_rows)} new entries to existing data.")
                        else:
                            combined_df = pd.concat([existing_df, df], ignore_index=True)
                            st.success(f"Added {len(df)} new entries to existing data.")
                        df = combined_df
                    
                    st.markdown("### Extracted Packages")
                    st.dataframe(df)
                    
                    # Create an Excel file in memory for download.
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False)
                    output.seek(0)
                    st.download_button(
                        "Download Excel", 
                        data=output, 
                        file_name="extracted_packages.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No packages could be extracted from the documents. Check if the documents contain the expected information.")
            
            # Clear button
            if st.button("Clear Extraction Documents"):
                st.session_state.documents_for_extraction = []
                st.rerun()

# ---------- Tab 3: General Document Assistant ----------
with tab3:
    st.header("General Document Assistant")
    st.write("This assistant can handle any document type and maintains full context for all your uploaded documents.")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Upload Documents")
        
        # Add option to select document source
        upload_option = st.radio(
            "Document source",
            ["Individual Files", "Folder Structure (ZIP)", "Local Folder"],
            key="general_upload_option"
        )
        
        if upload_option == "Individual Files":
            # File uploader for individual documents
            uploaded_files = st.file_uploader(
                "Upload documents to chat with", 
                type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="general_files"
            )
            
            # Process uploaded files
            if uploaded_files:
                with st.spinner("Processing your documents..."):
                    temp_dir = tempfile.mkdtemp()
                    docs = []
                    
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        docs.extend(load_document(file_path))
                    
                    if docs:
                        # Set up general RAG with all context enabled
                        st.session_state.general_qa_chain = setup_enhanced_rag_with_documents(docs, use_all_context=True)
                        st.success(f"{len(docs)} document(s) processed successfully!")
                    else:
                        st.error("No content could be extracted from the uploaded documents.")
        
        elif upload_option == "Folder Structure (ZIP)":
            # Add uploader for ZIP files containing the folder structure
            uploaded_zip = st.file_uploader(
                "Upload ZIP with folder structure", 
                type=["zip"],
                key="general_zip"
            )
            
            if uploaded_zip:
                with st.spinner("Processing your documents from ZIP..."):
                    try:
                        docs = extract_zip_and_process(uploaded_zip)
                        
                        if docs:
                            # Set up general RAG with all context enabled
                            st.session_state.general_qa_chain = setup_enhanced_rag_with_documents(docs, use_all_context=True)
                            
                            # Display document summary
                            st.success(f"{len(docs)} document(s) processed successfully!")
                            
                            # Group documents by type for better summary
                            doc_types = {}
                            for doc in docs:
                                doc_type = doc.metadata.get('doc_type', 'unknown')
                                if doc_type not in doc_types:
                                    doc_types[doc_type] = 0
                                doc_types[doc_type] += 1
                            
                            st.write("Document type summary:")
                            for doc_type, count in doc_types.items():
                                st.write(f"- {doc_type}: {count} document(s)")
                        else:
                            st.error("No content could be extracted from the ZIP file.")
                    except Exception as e:
                        st.error(f"Error processing ZIP file: {str(e)}")
        
        elif upload_option == "Local Folder":
            # Input for local folder path
            folder_path = st.text_input("Enter path to documents folder", key="general_folder")
            process_button = st.button("Process Folder", key="general_process")
            
            if folder_path and process_button:
                if os.path.isdir(folder_path):
                    with st.spinner("Processing documents from folder..."):
                        try:
                            docs = process_folder_structure(folder_path)
                            
                            if docs:
                                # Set up general RAG with all context enabled
                                st.session_state.general_qa_chain = setup_enhanced_rag_with_documents(docs, use_all_context=True)
                                st.success(f"{len(docs)} document(s) processed successfully!")
                            else:
                                st.error("No content could be extracted from the folder.")
                        except Exception as e:
                            st.error(f"Error processing folder: {str(e)}")
                else:
                    st.error("Invalid folder path. Please provide a valid directory path.")
        
        # Document statistics and clear buttons
        if st.session_state.all_documents:
            st.write(f"Total documents loaded: {len(st.session_state.all_documents)}")
            
            # Count document types
            doc_types = {}
            for doc in st.session_state.all_documents:
                doc_type = doc.metadata.get('doc_type', 'Unknown Type')
                if doc_type not in doc_types:
                    doc_types[doc_type] = 0
                doc_types[doc_type] += 1
            
            if doc_types:
                st.write("##### Document Types:")
                for doc_type, count in doc_types.items():
                    st.write(f"- {doc_type}: {count}")
        
        # Clear buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.session_state.general_qa_chain:
                if st.button("Clear Documents", key="general_clear_docs"):
                    st.session_state.general_qa_chain = None
                    st.session_state.all_documents = []
                    st.rerun()
        
        with col_b:
            if st.session_state.general_chat_history:
                if st.button("Clear Chat", key="general_clear_chat"):
                    st.session_state.general_chat_history = []
                    if st.session_state.general_qa_chain and hasattr(st.session_state.general_qa_chain, "memory"):
                        st.session_state.general_qa_chain.memory.clear()
                    st.rerun()
    
    with col1:
        # Chat interface
        st.subheader("Chat with All Documents")
        
        # Display document information
        if st.session_state.all_documents:
            st.info(f"You can ask questions about {len(st.session_state.all_documents)} documents across all uploads.")
            st.write("---")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.general_chat_history):
                if i % 2 == 0:
                    st.markdown(f"**You:** {message}")
                else:
                    st.markdown(f"**Assistant:** {message}")
        
        # Function to clear input after submission
        def submit_general_question():
            if st.session_state.general_user_input:
                st.session_state.general_user_question = st.session_state.general_user_input
                st.session_state.general_user_input = ""  # Clear the input field
        
        # User input with text area and submit button
        st.text_area("Ask a question about all your documents:", 
                     key="general_user_input", 
                     height=100,
                     value=st.session_state.general_user_input)
        
        if st.button("Submit", key="general_submit", on_click=submit_general_question):
            pass  # The actual action happens in the on_click callback
        
        # Process the question if there is one in the session state
        if st.session_state.general_user_question:
            if not st.session_state.general_qa_chain:
                st.warning("Please upload documents first.")
                st.session_state.general_user_question = ""  # Clear the question
            else:
                with st.spinner("Thinking..."):
                    try:
                        # Store the question first before processing
                        question = st.session_state.general_user_question
                        st.session_state.general_chat_history.append(question)
                        
                        # Use invoke with the question
                        response = st.session_state.general_qa_chain.invoke({
                            "question": question
                        })
                        
                        # Extract the answer
                        answer = response.get("answer", "I'm sorry, I couldn't generate a response.")
                        source_docs = response.get("source_documents", [])
                        
                        # Get source information to display
                        sources_info = []
                        seen_sources = set()
                        for doc in source_docs:
                            source = doc.metadata.get('source_file', 'Unknown')
                            doc_type = doc.metadata.get('doc_type', 'Unknown')
                            po = doc.metadata.get('po_number', '')
                            sf = doc.metadata.get('sf_code', '')
                            
                            source_key = source
                            if po and sf:
                                source_key = f"{source} (PO_{po}_SF_{sf}, {doc_type})"
                            elif doc_type != 'Unknown':
                                source_key = f"{source} ({doc_type})"
                            
                            if source_key not in seen_sources:
                                sources_info.append(source_key)
                                seen_sources.add(source_key)
                        
                        # Add sources to the answer if available
                        if sources_info:
                            sources_text = "\n\n*Sources: " + ", ".join(sources_info) + "*"
                            answer += sources_text
                        
                        # Add to chat history
                        st.session_state.general_chat_history.append(answer)
                        
                        # Clear the question to prevent reprocessing
                        st.session_state.general_user_question = ""
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        # Remove the question from history if there was an error
                        if st.session_state.general_chat_history and st.session_state.general_chat_history[-1] == question:
                            st.session_state.general_chat_history.pop()
                        st.session_state.general_user_question = ""
