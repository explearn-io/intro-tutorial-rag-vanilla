"""
File: plotembeddings.py
Description: [Brief description of the file's purpose]
Author: Arturo Gomez-Chavez
Creation Date: 07.07.2025
Institution/Organization: NA
Contributors/Editors:
License: MIT License - See LICENSE.MD file for details
Contact & Support:
- Email: [support@example.com]
"""

"""
plotembeddings.py - Embedding visualization and dimensionality reduction

This module provides functions for:
- Subsampling large embedding datasets for visualization
- Creating UMAP and t-SNE visualizations of document embeddings
- Generating interactive plots to explore document relationships
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import traceback
from sklearn.manifold import TSNE
from umap import UMAP


def subsample_chunks_per_document(embeddings_data, max_chunks_per_doc):
    """
    Subsample chunks per document to the specified maximum number.
    
    This function is important for visualization because:
    - Large numbers of chunks can make plots cluttered and slow
    - Uniform sampling ensures fair representation across documents
    - Reduces computational load for dimensionality reduction algorithms
    
    Args:
        embeddings_data (list): List of embedding data dictionaries, each containing:
                               - document_name: Name of the source document
                               - embedding: High-dimensional vector representation
                               - chunk_text: Original text content
        max_chunks_per_doc (int): Maximum number of chunks to keep per document
        
    Returns:
        list: Subsampled embedding data with uniform distribution across documents
    """
    # Group data by document name to ensure fair sampling
    doc_groups = {}
    for item in embeddings_data:
        doc_name = item['document_name']
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(item)
    
    # Subsample each document independently
    subsampled_data = []
    for doc_name, chunks in doc_groups.items():
        if len(chunks) <= max_chunks_per_doc:
            # If document has fewer chunks than max, keep all
            subsampled_data.extend(chunks)
        else:
            # Uniformly subsample chunks across the document
            # This ensures we get representative samples from beginning, middle, and end
            indices = np.linspace(0, len(chunks) - 1, max_chunks_per_doc, dtype=int)
            subsampled_chunks = [chunks[i] for i in indices]
            subsampled_data.extend(subsampled_chunks)
    
    return subsampled_data


def create_embedding_visualization(visualization_type, dimensions, max_chunks_per_doc):
    """
    Create UMAP or t-SNE visualization of document embeddings.
    
    This function performs dimensionality reduction to visualize high-dimensional
    embeddings in 2D or 3D space. This helps users understand:
    - How similar documents cluster together
    - The relationship between different document sections
    - The diversity of content in their knowledge base
    
    Args:
        visualization_type (str): Either 'UMAP' or 't-SNE'
                                 - UMAP: Better for preserving global structure, faster
                                 - t-SNE: Better for local neighborhoods, more detailed clusters
        dimensions (str): Either '2D' or '3D'
                         - 2D: Easier to interpret, better for presentations
                         - 3D: More detailed view, interactive exploration
        max_chunks_per_doc (int): Maximum number of chunks per document to visualize
        
    Returns:
        tuple: (plotly.graph_objects.Figure, dict) or (None, None) if error
               - Figure: Interactive plot for display
               - dict: Statistics about the visualization (chunk counts, dimensions, etc.)
    """
    try:
        # Step 1: Validate that embedding data exists
        if "full_embeddings_data" not in st.session_state:
            st.warning("❌ No embedding data found in session state. Please upload and process documents first.")
            return None, None
            
        if not st.session_state.full_embeddings_data:
            st.warning("❌ Embedding data is empty. Please upload and process documents first.")
            return None, None
        
        # Step 2: Subsample data for manageable visualization
        data_to_visualize = subsample_chunks_per_document(
            st.session_state.full_embeddings_data, 
            max_chunks_per_doc
        )
        
        if len(data_to_visualize) == 0:
            st.warning("❌ No data to visualize after subsampling.")
            return None, None
        
        # Step 3: Extract and prepare embedding data
        try:
            # Convert embeddings to numpy array for processing
            embedding_list = []
            for item in data_to_visualize:
                embedding = item['embedding']
                # Ensure embedding is a numpy array
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                elif not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                embedding_list.append(embedding)
            
            # Stack all embeddings into a matrix
            embeddings = np.vstack(embedding_list)
            
            # Extract metadata for plotting
            document_names = [item['document_name'] for item in data_to_visualize]
            chunk_texts = [item['chunk_text'][:100] + "..." if len(item['chunk_text']) > 100 
                           else item['chunk_text'] for item in data_to_visualize]
                           
        except Exception as e:
            st.error(f"❌ Error extracting/converting embeddings: {str(e)}")
            return None, None
        
        # Step 4: Validate embedding data
        if embeddings.size == 0:
            st.error("❌ Embeddings array is empty")
            return None, None
            
        if len(embeddings.shape) != 2:
            st.error(f"❌ Invalid embedding shape: {embeddings.shape}. Expected 2D array.")
            return None, None
        
        # Step 5: Set up dimensionality reduction parameters
        n_components = 2 if dimensions == '2D' else 3
        
        # Check minimum data requirements
        min_samples_needed = max(n_components + 1, 4)
        if len(embeddings) < min_samples_needed:
            st.error(f"❌ Need at least {min_samples_needed} data points for {visualization_type} {dimensions}. Found {len(embeddings)} points.")
            return None, None
        
        # Step 6: Apply dimensionality reduction
        try:
            if visualization_type == 'UMAP':
                # UMAP parameters optimized for document embeddings
                n_neighbors = min(15, len(embeddings) - 1)
                if n_neighbors < 2:
                    n_neighbors = 2
                reducer = UMAP(
                    n_components=n_components, 
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric='cosine'  # Good for text embeddings
                )
                reduced_embeddings = reducer.fit_transform(embeddings)
            else:  # t-SNE
                # t-SNE parameters optimized for text data
                perplexity = min(30, max(5, len(embeddings) - 1))
                if perplexity < 1:
                    perplexity = 1
                reducer = TSNE(
                    n_components=n_components, 
                    random_state=42, 
                    perplexity=perplexity,
                    max_iter=1000,
                    learning_rate='auto'
                )
                reduced_embeddings = reducer.fit_transform(embeddings)
        except Exception as e:
            st.error(f"❌ Error during {visualization_type} computation: {str(e)}")
            return None, None
        
        # Step 7: Create DataFrame for plotting
        try:
            plot_data = {
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'document': document_names,
                'chunk_text': chunk_texts
            }
            
            if n_components == 3:
                plot_data['z'] = reduced_embeddings[:, 2]
            
            df_plot = pd.DataFrame(plot_data)
        except Exception as e:
            st.error(f"❌ Error creating plot data: {str(e)}")
            return None, None
        
        # Step 8: Create interactive plot
        try:
            if dimensions == '2D':
                fig = px.scatter(
                    df_plot, 
                    x='x', 
                    y='y', 
                    color='document',
                    hover_data=['chunk_text'],
                    title=f'{visualization_type} {dimensions} Visualization of Document Embeddings',
                    labels={'x': f'{visualization_type} 1', 'y': f'{visualization_type} 2'},
                    width=800,
                    height=600
                )
            else:  # 3D
                fig = px.scatter_3d(
                    df_plot, 
                    x='x', 
                    y='y', 
                    z='z',
                    color='document',
                    hover_data=['chunk_text'],
                    title=f'{visualization_type} {dimensions} Visualization of Document Embeddings',
                    labels={'x': f'{visualization_type} 1', 'y': f'{visualization_type} 2', 'z': f'{visualization_type} 3'},
                    width=800,
                    height=600
                )
        except Exception as e:
            st.error(f"❌ Error creating plot: {str(e)}")
            return None, None
        
        # Step 9: Enhance plot appearance and interactivity
        try:
            fig.update_layout(
                legend_title="Documents",
                showlegend=True,
                font=dict(size=12),
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            # Create informative hover template that uses the data correctly
            # Since Plotly creates separate traces for each document when using color='document',
            # we need to use the built-in hover data instead of customdata
            if dimensions == '2D':
                hovertemplate = ('<b>Document:</b> %{fullData.name}<br>' +
                               '<b>Chunk Text:</b> %{customdata[0]}<br>' +
                               '<extra></extra>')
            else:
                hovertemplate = ('<b>Document:</b> %{fullData.name}<br>' +
                               '<b>Chunk Text:</b> %{customdata[0]}<br>' +
                               '<extra></extra>')
            
            # Alternative approach: Use the color data directly and update each trace individually
            # This is more reliable than trying to use customdata across multiple traces
            for i, trace in enumerate(fig.data):
                # Get the document name for this trace
                trace_name = trace.name
                
                # Find all points belonging to this document
                trace_indices = [j for j, doc in enumerate(document_names) if doc == trace_name]
                trace_chunk_texts = [chunk_texts[j] for j in trace_indices]
                
                # Update hover template for this specific trace
                if dimensions == '2D':
                    trace_hovertemplate = (f'<b>Document:</b> {trace_name}<br>' +
                                         '<b>Chunk Text:</b> %{customdata}<br>' +
                                         '<extra></extra>')
                else:
                    trace_hovertemplate = (f'<b>Document:</b> {trace_name}<br>' +
                                         '<b>Chunk Text:</b> %{customdata}<br>' +
                                         '<extra></extra>')
                
                # Update this trace with its specific hover data
                fig.data[i].update(
                    hovertemplate=trace_hovertemplate,
                    customdata=trace_chunk_texts
                )
        except Exception as e:
            st.error(f"❌ Error updating plot layout: {str(e)}")
            return None, None
        
        # Step 10: Calculate and return statistics
        try:
            doc_chunk_counts = {}
            for item in data_to_visualize:
                doc_name = item['document_name']
                doc_chunk_counts[doc_name] = doc_chunk_counts.get(doc_name, 0) + 1
            
            stats = {
                'total_points': len(data_to_visualize),
                'doc_counts': doc_chunk_counts,
                'embedding_dim': embeddings.shape[1],
                'reduced_dim': n_components
            }
        except Exception as e:
            st.error(f"❌ Error calculating statistics: {str(e)}")
            return None, None
        
        return fig, stats
        
    except Exception as e:
        st.error(f"❌ Unexpected error in visualization: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())
        return None, None