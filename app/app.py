"""
app.py - Main Streamlit application for RAG-based document query system

Description: [Brief description of the file's purpose]
Author: Arturo Gomez-Chavez
Creation Date: 07.07.2025
Institution/Organization: NA
Contributors/Editors:
License: MIT License - See LICENSE.MD file for details
Contact & Support:
- Email: [support@example.com]

This is the main application file that brings together all RAG components:
- Document processing and chunking (dbchunks.py)
- Embedding visualization (plotembeddings.py) 
- RAG chain for question answering (ragchain.py)
- Utility functions for model management (utils.py)

For RAG beginners, this file demonstrates how different components work together
to create a complete document-based question answering system.
"""

import streamlit as st
import pandas as pd
import time
import traceback

# Import our custom modules
from utils import initialize_models, clear_models, reset_database, get_safe_chunking_defaults, validate_chunking_parameters
from dbchunks import chunkize_and_add_to_db, get_chunk_statistics
from plotembeddings import create_embedding_visualization
from ragchain import run_rag_chain, get_relevant_documents, validate_prompt_template, validate_temperature, get_temperature_description


###>>> START OF SIDEBAR SECTION <<<###
def render_sidebar():
    """
    Renders the sidebar with API key management, document processing settings,
    and database management controls.
    
    Returns:
        tuple: (chunk_size, chunk_overlap, pdf_docs) - Current chunking parameters and uploaded files
    """
    with st.sidebar:
        st.title("🔑 Configuration")
        
        # API Key Management Section
        api_key = render_api_key_section()
        
        st.markdown("---")
        
        # Document Processing Settings Section
        chunk_size, chunk_overlap, pdf_docs = render_document_processing_section()
        
        st.markdown("---")
        
        # Database Management Section
        render_database_management_section()
    
    return chunk_size, chunk_overlap, pdf_docs


def render_api_key_section():
    """
    Renders the API key management section of the sidebar.
    
    Returns:
        str: The current Gemini API key
    """
    st.subheader("API Key Setup")
    gemini_api_key = st.text_input(
        "Enter your Gemini API key:", 
        type="password",
        help="Get your API key from Google AI Studio"
    )

    if st.button("💾 Save API Key"):
        if gemini_api_key:
            # Check if API key has changed, if so clear cached models
            if st.session_state.get("gemini_api_key") != gemini_api_key:
                clear_models()
            
            st.session_state.gemini_api_key = gemini_api_key
            st.success("✅ API key saved!")
        else:
            st.warning("⚠️ Please enter your Gemini API key to proceed.")
    
    return gemini_api_key


def render_document_processing_section():
    """
    Renders the document processing section with chunking parameters and file uploader.
    
    Returns:
        tuple: (chunk_size, chunk_overlap, pdf_docs) - Current settings and uploaded files
    """
    st.subheader("📄 Document Processing")
    
    # Show current chunking parameters if they exist
    if st.session_state.get("chunking_params_set"):
        st.info(
            f"🔧 **Current Settings:**\n"
            f"- Chunk Size: {st.session_state.get('current_chunk_size', 'N/A')}\n"
            f"- Chunk Overlap: {st.session_state.get('current_chunk_overlap', 'N/A')}"
        )
        st.warning("⚠️ Chunking parameters are locked after uploading files. Use 'Reset Database' to change them.")
    
    # Get safe default values to prevent widget errors
    default_chunk_size = st.session_state.get('current_chunk_size', 200)
    default_chunk_overlap = st.session_state.get('current_chunk_overlap', 50)
    safe_chunk_size, safe_overlap = get_safe_chunking_defaults(default_chunk_size, default_chunk_overlap)
    
    # Chunking parameters (disabled if already set)
    chunk_size = st.number_input(
        "Chunk Size (tokens):",
        min_value=50,
        max_value=1000,
        value=safe_chunk_size,
        step=50,
        help="Number of tokens per chunk. Larger chunks provide more context but may be less precise.",
        disabled=st.session_state.get("chunking_params_set", False)
    )
    
    # Calculate safe defaults for chunk overlap
    max_overlap = min(chunk_size - 1, 200)
    current_overlap_value = safe_overlap
    
    # Ensure current value doesn't exceed the maximum allowed
    if current_overlap_value > max_overlap:
        current_overlap_value = max_overlap
    
    chunk_overlap = st.number_input(
        "Chunk Overlap (tokens):",
        min_value=0,
        max_value=max_overlap,
        value=current_overlap_value,
        step=min(25, max(1, max_overlap // 4)),  # Dynamic step size
        help="Number of overlapping tokens between chunks. Helps preserve context across boundaries.",
        disabled=st.session_state.get("chunking_params_set", False)
    )
    
    # Validation and user feedback
    render_chunking_validation(chunk_size, chunk_overlap)
    
    # File uploader
    pdf_docs = st.file_uploader(
        "📎 Upload your documents (PDF):",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF documents to create your knowledge base"
    )
    
    # Process documents button
    handle_document_processing(pdf_docs, chunk_size, chunk_overlap)
    
    return chunk_size, chunk_overlap, pdf_docs


def render_chunking_validation(chunk_size, chunk_overlap):
    """
    Renders validation feedback for chunking parameters.
    
    Args:
        chunk_size (int): The chunk size to validate
        chunk_overlap (int): The chunk overlap to validate
    """
    validation = validate_chunking_parameters(chunk_size, chunk_overlap)
    
    # Display errors
    for error in validation['errors']:
        st.error(f"❌ {error}")
    
    # Display warnings
    for warning in validation['warnings']:
        st.warning(f"⚠️ {warning}")
    
    # Show current settings summary
    if not st.session_state.get("chunking_params_set", False):
        with st.expander("📊 Current Settings Summary", expanded=False):
            for info in validation['info']:
                st.write(f"• {info}")
            
            if validation['is_valid']:
                st.success("✅ Parameters are valid and ready for processing!")
            else:
                st.error("❌ Please fix the errors above before processing documents.")


def handle_document_processing(pdf_docs, chunk_size, chunk_overlap):
    """
    Handles the document processing workflow when the user clicks the process button.
    
    Args:
        pdf_docs (list): List of uploaded PDF files
        chunk_size (int): Chunk size for processing
        chunk_overlap (int): Chunk overlap for processing
    """
    if st.button("🚀 Process Documents", type="primary"):
        validation = validate_chunking_parameters(chunk_size, chunk_overlap)
        
        if not pdf_docs:
            st.warning("⚠️ Please upload at least one PDF file")
        elif not st.session_state.get("gemini_api_key"):
            st.warning("⚠️ Please enter your Gemini API key first!")
        elif not validation['is_valid']:
            st.error("⚠️ Please fix the chunking parameter errors before processing!")
        else:
            with st.spinner("🔄 Processing your documents..."):
                chunkize_and_add_to_db(pdf_docs, chunk_size, chunk_overlap)


def render_database_management_section():
    """
    Renders the database management section with statistics and reset functionality.
    """
    st.subheader("🗃️ Database Management")
    
    # Show current database statistics
    stats = get_chunk_statistics()
    if stats["total_chunks"] > 0:
        st.metric("Documents", stats["total_documents"])
        st.metric("Total Chunks", stats["total_chunks"])
    else:
        st.info("📝 No documents processed yet")
    
    # Reset database button
    if st.button("🗑️ Reset Database", type="secondary"):
        if reset_database():
            st.success("✅ Database reset successfully! You can now upload new documents.")
            st.rerun()
        else:
            st.error("❌ Failed to reset database. Please try again.")

###>>> END OF SIDEBAR SECTION <<<###

###>>> START OF RAG QUERY TAB <<<###
def render_query_tab():
    """
    Renders the main query interface tab.
    """
    st.subheader("💬 Ask Questions About Your Documents")
    
    # System status indicators
    render_system_status()
    st.markdown("---")

    # Advanced Configuration Section
    st.subheader("🕵🏽 Customize AI Assistant")
    
    # Temperature Control Section
    temperature = render_temperature_control()
    st.markdown("---")

    # Prompt customization section
    custom_prompt, default_prompt = render_prompt_customization()

    # Query interface
    st.subheader("🔍 Ask Your Question")
    query = st.text_area(
        "What would you like to know about your documents?",
        placeholder="e.g., What are the main findings about climate change impacts?\ne.g., Summarize the key recommendations from the reports.",
        height=100
    )

    # Submit query button
    if st.button("🚀 Get Answer", type="primary"):
        handle_query_submission(query, custom_prompt, default_prompt)

    # Example queries for beginners
    stats = get_chunk_statistics()
    if stats["total_chunks"] == 0:
        st.info(
            "📚 **No documents uploaded yet!**\n\n"
            "Upload some PDF documents using the sidebar to start asking questions. "
            "Once you have documents, you can ask questions like:\n"
            "- 'What are the main topics covered?'\n"
            "- 'Summarize the key findings'\n"
            "- 'What recommendations are provided?'"
        )


def render_system_status():
    """
    Renders system status indicators showing the current state of the application.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.get("gemini_api_key"):
            st.success("✅ API Key")
        else:
            st.error("❌ API Key")
    
    with col2:
        if st.session_state.get("db"):
            st.success("✅ Database")
        else:
            st.warning("⚠️ Database")
    
    with col3:
        stats = get_chunk_statistics()
        if stats["total_documents"] > 0:
            st.success(f"✅ {stats['total_documents']} Docs")
        else:
            st.warning("⚠️ No Docs")
    
    with col4:
        if stats["total_chunks"] > 0:
            st.info(f"📑 {stats['total_chunks']} Chunks")
        else:
            st.warning("📑 0 Chunks")


def render_temperature_control():
    """
    Renders the temperature control section for AI response creativity.
    
    Returns:
        float: The selected temperature value
    """
    st.subheader("🌡️ Response Creativity Control")
    
    # Initialize temperature in session state if not present
    if "temperature" not in st.session_state:
        st.session_state.temperature = 1.0
    
    # Temperature slider with real-time feedback
    temperature = st.slider(
        "Temperature: Common LLM parameter that controls randomness/creativity of responses",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get("temperature", 1.0),
        step=0.1,
        help="Controls randomness in AI responses. Lower = more focused, Higher = more creative"
    )
    
    # Store temperature in session state
    st.session_state.temperature = temperature
    
    # Validate and show temperature description
    is_valid, error_msg, normalized_temp = validate_temperature(temperature)
    if is_valid:
        description = get_temperature_description(temperature)
        st.markdown(description)
    else:
        st.error(f"⚠️ {error_msg}")
    
    # Temperature examples and guidance
    with st.expander("🎯 Temperature Guide & Examples", expanded=False):
        st.write(
            "**Temperature Examples:**\n\n"
            "🎯 **0.0 (Deterministic)** - Perfect for:\n"
            "- Factual questions requiring consistent answers\n"
            "- Technical documentation queries\n"
            "- When you need the same response every time\n\n"
            
            "⚖️ **0.7 (Balanced)** - Great for:\n"
            "- General knowledge questions\n"
            "- Explanations that benefit from natural variation\n"
            "- Most everyday use cases\n\n"
            
            "🎨 **1.0 (Default - Creative)** - Ideal for:\n"
            "- Complex analysis requiring interpretation\n"
            "- Brainstorming and idea generation\n"
            "- Natural, conversational responses\n\n"
            
            "🚀 **1.5+ (Highly Creative)** - Best for:\n"
            "- Creative writing tasks\n"
            "- Exploring multiple perspectives\n"
            "- When you want diverse, unexpected insights\n\n"
            
            "💡 **Tip:** Start with 1.0 and adjust based on your needs!"
        )
    
    # Current temperature status and quick presets
    render_temperature_presets(temperature)
    
    return temperature


def render_temperature_presets(temperature):
    """
    Renders temperature status and quick preset buttons.
    
    Args:
        temperature (float): Current temperature value
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric("**Current Temperature**", f"{temperature:.1f}")
    with col2:
        # Quick preset buttons
        st.write("**Quick Presets:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button("🎯 Factual", help="Temperature: 0.0"):
                st.session_state.temperature = 0.0
                st.rerun()
        with preset_col2:
            if st.button("⚖️ Balanced", help="Temperature: 0.7"):
                st.session_state.temperature = 0.7
                st.rerun()
        with preset_col3:
            if st.button("🎨 Creative", help="Temperature: 1.0"):
                st.session_state.temperature = 1.0
                st.rerun()


def render_prompt_customization():
    """
    Renders the prompt customization section.
    
    Returns:
        tuple: (custom_prompt, default_prompt) - The custom prompt and default prompt
    """
    # Default prompt template
    default_prompt = """You are a highly knowledgeable assistant specializing in Environmental sciences. 
Answer the question based only on the following context:
{context}

Answer the question based on the above context:
{question}

Use the provided context to answer the user's question accurately and concisely.
Justify your answers.
Don't give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar."""

    # Prompt customization section
    with st.expander("🔧 Advanced: Custom Prompt Template", expanded=False):
        st.info(
            "💡 **Prompt Engineering Tips:**\n"
            "- Your template must include `{context}` and `{question}` placeholders\n"
            "- Specify the domain/field of expertise\n"
            "- Set the tone and response style\n"
            "- Add specific instructions for handling edge cases"
        )
        
        custom_prompt = st.text_area(
            "Custom prompt template:",
            value=default_prompt,
            height=200,
            help="Modify this template to change how the AI responds to questions"
        )
        
        # Validate prompt template
        is_valid, error_msg = validate_prompt_template(custom_prompt)
        if not is_valid:
            st.error(f"⚠️ {error_msg}")
        else:
            st.success("✅ Prompt template is valid")
        
        if st.button("🔄 Reset to Default Prompt"):
            st.rerun()
    
    return custom_prompt, default_prompt


def handle_query_submission(query, custom_prompt, default_prompt):
    """
    Handles the query submission and response generation.
    
    Args:
        query (str): The user's question
        custom_prompt (str): The custom prompt template
        default_prompt (str): The default prompt template
    """
    if not query.strip():
        st.warning("⚠️ Please enter a question")
    elif not st.session_state.get("gemini_api_key"):
        st.warning("⚠️ Please enter your Gemini API key first!")
    elif not st.session_state.get("db"):
        st.warning("⚠️ Please upload and process documents first!")
    else:
        # Validate prompt template before processing
        is_valid, error_msg = validate_prompt_template(custom_prompt)
        if not is_valid:
            st.error(f"⚠️ Fix your prompt template: {error_msg}")
        else:
            with st.spinner("🤔 Thinking and searching through your documents..."):
                # Get the current temperature from session state
                current_temperature = st.session_state.get("temperature", 1.0)
                
                # Get the answer using RAG with temperature control
                result = run_rag_chain(
                    query=query, 
                    custom_prompt=custom_prompt,
                    temperature=current_temperature
                )
                
                # Display the result
                st.subheader("📝 Answer:")
                st.write(result)
                
                # Show additional details
                render_answer_details(query, current_temperature, custom_prompt, default_prompt)


def render_answer_details(query, current_temperature, custom_prompt, default_prompt):
    """
    Renders detailed information about the generated answer.
    
    Args:
        query (str): The user's question
        current_temperature (float): Temperature used for generation
        custom_prompt (str): The custom prompt used
        default_prompt (str): The default prompt for comparison
    """
    with st.expander("🔍 Details About This Answer", expanded=False):
        st.write(f"**Your Question:** {query}")
        st.write(f"**Temperature Used:** {current_temperature}")
        st.write(f"**Temperature Effect:** {get_temperature_description(current_temperature)}")
        st.write(f"**Custom Prompt Used:** {'Yes' if custom_prompt.strip() != default_prompt.strip() else 'No (using default)'}")
        
        # Try to show retrieved documents
        try:
            docs = get_relevant_documents(query, k=5)
            if docs:
                st.write(f"**Retrieved {len(docs)} relevant document chunks (only showing top 3):**")
                for i, doc in enumerate(docs[:3]):  # Show first 3
                    st.write(f"**Chunk {i+1}:** {doc.page_content}...")
            else:
                st.write("**No relevant documents found for this query**")
        except Exception as e:
            st.write(f"Could not retrieve document details: {str(e)}")

###>>> END OF RAG QUERY TAB <<<###

###>>> START OF EMBEDDING VISUALIZATION TAB <<<###
def render_embedding_tab():
    """
    Renders the embedding visualization tab.
    """
    st.subheader("📊 Explore Your Document Embeddings")
    
    # Check if embedding data exists
    if "full_embeddings_data" in st.session_state and st.session_state.full_embeddings_data:
        # Get document count for reference
        doc_names = list(set([item['document_name'] for item in st.session_state.full_embeddings_data]))
        total_chunks = len(st.session_state.full_embeddings_data)
        
        st.info(f"📚 **Available Data:** {len(doc_names)} documents, {total_chunks} total chunks")
        
        # Document Information Table
        render_document_info_table()
        st.markdown("---")
        
        # Visualization controls and generation
        viz_type, dimensions, max_chunks = render_visualization_controls()
        handle_visualization_generation(viz_type, dimensions, max_chunks)
    
    else:
        # No embedding data available
        st.info("📋 **No embedding data available yet.**")
        st.write(
            "Upload and process documents using the sidebar to generate embeddings for visualization.\n\n"
            "**What you'll see here after uploading documents:**\n"
            "- Interactive UMAP or t-SNE plots showing document relationships\n"
            "- 2D and 3D visualization options\n"
            "- Hover details showing chunk content\n"
            "- Statistics about your document collection\n"
            "- Insights into how your documents cluster by topic"
        )
    
        # Show example of what visualization looks like
        st.write("**Example Visualization:**")
        st.image("./example-plot-embeddings.png", use_container_width=True,
                 caption="Example: Document chunks visualized in 2D space, colored by source document")

def render_document_info_table():
    """
    Renders the document information table if available.
    """
    if "document_info" in st.session_state and st.session_state.document_info:
        st.subheader("📋 Document Chunks Preview")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(st.session_state.document_info)
        
        # Display the table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Document Name": st.column_config.TextColumn("Document", width="medium"),
                "Chunk Index": st.column_config.NumberColumn("Chunk #", width="small"),
                "Chunk Text": st.column_config.TextColumn("Content Preview", width="large"),
                "Embedding": st.column_config.TextColumn("Vector Preview", width="large"),
                "Embedding Length": st.column_config.NumberColumn("Vector Dims", width="small"),
                "Chunk Size": st.column_config.NumberColumn("Chunk Size", width="small"),
                "Chunk Overlap": st.column_config.NumberColumn("Overlap", width="small")
            }
        )
        
        st.caption(f"📋 Preview showing up to 5 chunks per document. Total displayed: {len(df)}")

def render_visualization_controls():
    """
    Renders the visualization controls and handles visualization generation.
    
    Returns:
        tuple: (viz_type, dimensions, max_chunks) - Visualization parameters
    """
    st.subheader("🎨 Create Embedding Visualization")
    st.write("Visualize how your document chunks relate to each other in high-dimensional space!")

    # Interpretation guide
    with st.expander("📖 How to Interpret This Visualization", expanded=False):
        st.write(
            "**What you're seeing:**\n"
            "- Each point represents a chunk from your documents\n"
            "- Points close together have similar content\n"
            "- Different colors represent different documents\n"
            "- Clusters indicate topics or themes\n\n"
            "**What this tells you:**\n"
            "- How diverse your document collection is\n"
            "- Which documents are most similar\n"
            "- Whether topics are well-separated or overlapping\n"
            "- How the RAG system might group related information"
        )

    # Debug information for advanced users
    render_debug_information()
    
    # Create controls for the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        viz_type = st.selectbox(
            "🔬 Algorithm:",
            ["UMAP", "t-SNE"],
            help="UMAP: Faster, preserves global structure\nt-SNE: Better local neighborhoods"
        )
    
    with col2:
        dimensions = st.selectbox(
            "📐 Dimensions:",
            ["2D", "3D"],
            help="2D: Easier to interpret\n3D: More detailed exploration"
        )
    
    with col3:
        max_chunks = st.number_input(
            "📊 Max chunks per doc:",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Higher values = more detail but slower computation"
        )
    
    return viz_type, dimensions, max_chunks

def render_debug_information():
    """
    Renders debug information for advanced users.
    """
    with st.expander("🔧 Advanced: Debug Information", expanded=False):
        if st.session_state.full_embeddings_data:
            sample_item = st.session_state.full_embeddings_data[0]
            st.write("**Example of document encoding:**")
            st.json({
                "document_name": sample_item.get('document_name', 'N/A'),
                "chunk_index": sample_item.get('chunk_index', 'N/A'),
                "embedding_length": len(sample_item.get('embedding', [])) if sample_item.get('embedding') else 0,
                "embedding_type": type(sample_item.get('embedding', None)).__name__,
                "chunk_text_length": len(sample_item.get('chunk_text', ''))
            })
            
            st.write("**Document distribution:**")
            doc_counts = {}
            for item in st.session_state.full_embeddings_data:
                doc_name = item.get('document_name', 'Unknown')
                doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
            st.write(doc_counts)

def handle_visualization_generation(viz_type, dimensions, max_chunks):
    """
    Handles the visualization generation process.
    
    Args:
        viz_type (str): Type of visualization (UMAP or t-SNE)
        dimensions (str): Dimensions (2D or 3D)
        max_chunks (int): Maximum chunks per document
    """
    if st.button("🎨 Generate Visualization", type="primary"):
        progress_container = st.container()
        plot_container = st.container()
        
        with progress_container:
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                status_placeholder.info("🚀 Initializing visualization...")
                progress_bar.progress(20)
                
                with st.spinner(f"Computing {viz_type} {dimensions} visualization..."):
                    progress_bar.progress(60)
                    status_placeholder.info(f"🧮 Running {viz_type} algorithm...")
                    
                    result = create_embedding_visualization(viz_type, dimensions, max_chunks)
                    progress_bar.progress(100)
                    
                    if result and result != (None, None):
                        fig, stats = result
                        
                        if fig is not None and stats is not None:
                            status_placeholder.success("✅ Visualization completed!")
                            time.sleep(1)
                            progress_container.empty()
                            
                            with plot_container:
                                # Display the plot
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            status_placeholder.error("❌ Visualization failed")
                    else:
                        status_placeholder.error("❌ Could not generate visualization")
                        
            except Exception as e:
                progress_bar.progress(100)
                status_placeholder.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

###>>> END OF EMBEDDING VISUALIZATION TAB <<<###
def main():
    """
    Main function that sets up and runs the Streamlit RAG application.
    
    This application demonstrates a complete RAG (Retrieval-Augmented Generation) workflow:
    1. Document Upload & Processing: Users upload PDFs which are split into chunks
    2. Vector Storage: Chunks are embedded and stored in a vector database
    3. Query Interface: Users ask questions about their documents
    4. Retrieval: System finds relevant document chunks
    5. Generation: Language model generates answers based on retrieved context
    6. Visualization: Users can explore document embeddings in 2D/3D space
    """
    
    # Configure the Streamlit page
    st.set_page_config(
        page_title="RAG Knowledge Query System", 
        page_icon="🔍",
        layout="wide"
    )
    st.header("🔍 RAG Knowledge Query System")
    st.subheader("Upload documents, ask questions, and explore embeddings!")

    # Render sidebar
    chunk_size, chunk_overlap, pdf_docs = render_sidebar()

    # Main content area with tabs
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Explore Embeddings"])

    # TAB 1: Query Interface
    with tab1:
        render_query_tab()

    # TAB 2: Embedding Visualization
    with tab2:
        render_embedding_tab()


if __name__ == "__main__":
    main()