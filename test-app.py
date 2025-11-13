"""
dbchunks.py - Document processing and database management

This module handles:
- Loading and processing PDF documents
- Splitting documents into smaller chunks
- Adding processed chunks to the vector database
- Managing document metadata and embeddings
- Cleaning up old databases before processing new documents
"""

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from utils import initialize_models, cleanup_old_databases


def chunkize_and_add_to_db(uploaded_files, chunk_size, chunk_overlap):
    """
    Processes uploaded PDF files by splitting them into chunks and adding to the database.
    
    This function performs the following steps:
    0. Cleans up old database directories to prevent accumulation
    1. Validates input files and API key
    2. Initializes models if needed
    3. For each PDF file:
       - Saves to temporary location
       - Loads content using PyPDFLoader
       - Splits content into smaller chunks
       - Generates embeddings for each chunk
       - Adds chunks to the vector database
       - Stores metadata for display and visualization
    4. Cleans up temporary files
    
    Args:
        uploaded_files (list): List of uploaded PDF file objects from Streamlit
        chunk_size (int): Maximum number of tokens per chunk
        chunk_overlap (int): Number of tokens to overlap between consecutive chunks
    
    Returns:
        None
        
    The chunking process is important because:
    - Large documents need to be broken into smaller pieces for better retrieval
    - Overlapping chunks ensure important information isn't lost at chunk boundaries
    - Smaller chunks allow for more precise similarity matching during retrieval
    """
    
    # Validate inputs
    if not uploaded_files:
        st.error("❌ No files uploaded!")
        return

    if not st.session_state.get("gemini_api_key"):
        st.error("❌ Please enter your Gemini API key first!")
        return

    # Step 0: Clean up old database directories before creating new ones
    st.info("🧹 Cleaning up old databases...")
    cleanup_old_databases()
    
    # Clear existing session state to start fresh
    session_keys_to_clear = [
        "db", 
        "embedding_model", 
        "document_info", 
        "full_embeddings_data"
    ]
    
    for key in session_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Initialize models (embedding model and vector database)
    st.info("🚀 Initializing fresh database...")
    initialize_models()

    # Initialize storage for document information if not exists
    if "document_info" not in st.session_state:
        st.session_state.document_info = []
    
    # Initialize storage for full embeddings (used for visualization)
    if "full_embeddings_data" not in st.session_state:
        st.session_state.full_embeddings_data = []

    # Store chunking parameters to ensure consistency across uploads
    st.session_state.chunking_params_set = True
    st.session_state.current_chunk_size = chunk_size
    st.session_state.current_chunk_overlap = chunk_overlap

    # Create progress bar and status placeholder
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each uploaded file
    for file_index, uploaded_file in enumerate(uploaded_files):
        # Update progress bar and status
        progress = (file_index) / total_files
        progress_bar.progress(progress)
        status_text.text(f"📄 Processing: {uploaded_file.name} ({file_index + 1}/{total_files})")
        
        # Step 1: Save uploaded file to temporary location
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        # Step 2: Load PDF content using LangChain's PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()  # Returns list of Document objects

        # Step 3: Extract metadata and content from loaded documents
        doc_metadata = [data[i].metadata for i in range(len(data))]
        doc_content = [data[i].page_content for i in range(len(data))]

        # Step 4: Split documents into smaller, manageable chunks
        # SentenceTransformersTokenTextSplitter ensures chunks respect token boundaries
        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Tokenizer model
            chunk_size=chunk_size,        # Maximum tokens per chunk
            chunk_overlap=chunk_overlap   # Overlapping tokens between chunks
        )
        st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)
        
        # Update status with chunk information
        status_text.text(f"📊 Processing {uploaded_file.name}: Created {len(st_chunks)} chunks ({file_index + 1}/{total_files})")

        # Step 5: Add all chunks to the vector database
        # This automatically generates embeddings and stores them
        st.session_state.db.add_documents(st_chunks)

        # Step 6: Generate and store embeddings for ALL chunks (for visualization)
        for i, chunk in enumerate(st_chunks):
            # Generate embedding vector for this chunk
            embedding = st.session_state.embedding_model.embed_query(chunk.page_content)
            
            # Store comprehensive embedding data for visualization purposes
            embedding_data = {
                "document_name": uploaded_file.name,
                "chunk_index": i,
                "chunk_text": chunk.page_content,
                "embedding": embedding,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            st.session_state.full_embeddings_data.append(embedding_data)

        # Step 7: Store sample chunk information for display table (limit to 5 per document)
        chunks_to_display = st_chunks[:5]  # Show only first 5 chunks in UI table
        
        for i, chunk in enumerate(chunks_to_display):
            # Generate embedding for display
            embedding = st.session_state.embedding_model.embed_query(chunk.page_content)
            
            # Create document info for the display table
            doc_info = {
                "Document Name": uploaded_file.name,
                "Chunk Index": i,
                "Chunk Text": chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content,
                "Embedding": str(embedding[:10]) + "..." if len(embedding) > 10 else str(embedding),  # Show first 10 dimensions
                "Embedding Length": len(embedding),
                "Chunk Size": chunk_size,
                "Chunk Overlap": chunk_overlap
            }
            st.session_state.document_info.append(doc_info)

        # Step 8: Clean up - remove temporary file
        os.remove(temp_file_path)
    
    # Complete the progress bar and show final status
    progress_bar.progress(1.0)
    status_text.text(f"✅ Successfully processed all {total_files} documents!")


def get_chunk_statistics():
    """
    Get statistics about the current chunks in the database.
    
    Returns:
        dict: Statistics including total chunks, documents, and chunk distribution
    """
    if "full_embeddings_data" not in st.session_state:
        return {"total_chunks": 0, "total_documents": 0, "chunks_per_document": {}}
    
    data = st.session_state.full_embeddings_data
    
    # Count chunks per document
    chunks_per_doc = {}
    for item in data:
        doc_name = item.get('document_name', 'Unknown')
        chunks_per_doc[doc_name] = chunks_per_doc.get(doc_name, 0) + 1
    
    return {
        "total_chunks": len(data),
        "total_documents": len(chunks_per_doc),
        "chunks_per_document": chunks_per_doc
    }