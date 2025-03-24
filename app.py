import streamlit as st
import logging
import json
import pandas as pd
import time
from datetime import datetime

from utils.document_processor import DocumentProcessor
from utils.search import SearchEngine
from utils.answer_formatter import AnswerFormatter
import config

# Set up logging
logger = logging.getLogger(__name__)

# Initialize components
@st.cache_resource
def initialize_components():
    search_engine = SearchEngine(
        openai_api_key=config.OPENAI_API_KEY,
        qdrant_url=config.QDRANT_URL,
        qdrant_api_key=config.QDRANT_API_KEY,
        collection_name=config.DEFAULT_COLLECTION,
        embedding_model=config.EMBEDDING_MODEL,
        vector_size=config.VECTOR_SIZE,
        llm_model=config.LLM_MODEL
    )
    
    document_processor = DocumentProcessor()
    answer_formatter = AnswerFormatter()
    
    return search_engine, document_processor, answer_formatter

# Function to log user activity
def log_activity(activity_type, details=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    activity = {
        "timestamp": timestamp,
        "type": activity_type,
        "details": details or {}
    }
    
    try:
        # Append to file
        with open("user_activity.log", "a") as f:
            f.write(json.dumps(activity) + "\n")
    except Exception as e:
        logger.error(f"Failed to log activity: {e}")

# Function to track usage metrics
def track_metric(metric_name, value=1):
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    
    if metric_name not in st.session_state.metrics:
        st.session_state.metrics[metric_name] = 0
    
    st.session_state.metrics[metric_name] += value

# Main app function
def main():
    # Initialize components
    search_engine, document_processor, answer_formatter = initialize_components()
    
    # App header
    st.title(config.APP_TITLE)
    st.markdown(config.APP_SUBTITLE)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        collection_name = st.text_input(
            "Collection Name",
            value=config.DEFAULT_COLLECTION
        )
        
        search_limit = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=config.DEFAULT_SEARCH_LIMIT
        )
        
        use_llm = st.checkbox("Use LLM for improved answers", value=True)
        
        st.header("About")
        st.markdown("""
        This RFP Q&A system helps you find answers to questions about RFP documents.
        Upload documents or ask questions about existing data.
        """)
        
        # Simple usage metrics
        if 'metrics' in st.session_state:
            st.header("Session Stats")
            for metric, count in st.session_state.metrics.items():
                st.text(f"{metric}: {count}")
    
    # Main layout with two columns
    col1, col2 = st.columns([1, 1])
    
    # Left column for input
    with col1:
        # Document upload tab
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document (PDF, DOCX, XLSX, TXT)",
            type=["pdf", "docx", "xlsx", "txt"]
        )
        
        if uploaded_file:
            try:
                with st.spinner("Processing document..."):
                    # Process the uploaded file
                    start_time = time.time()
                    extracted_text = document_processor.process_uploaded_file(uploaded_file)
                    processing_time = time.time() - start_time
                    
                    # Show preview of extracted text
                    st.subheader("Extracted Content Preview")
                    st.text_area(
                        "Preview (first 1000 characters)",
                        extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""),
                        height=150
                    )
                    
                    st.info(f"Document processed in {processing_time:.2f} seconds. Total characters: {len(extracted_text)}")
                    
                    # Chunk the text
                    chunks = document_processor.chunk_text(
                        extracted_text,
                        chunk_size=config.CHUNK_SIZE,
                        overlap=config.CHUNK_OVERLAP
                    )
                    
                    st.text(f"Document split into {len(chunks)} chunks")
                    
                    # Save to session state for indexing
                    st.session_state.extracted_text = extracted_text
                    st.session_state.chunks = chunks
                    
                    # Log activity
                    log_activity("document_upload", {
                        "filename": uploaded_file.name,
                        "size": len(extracted_text),
                        "chunks": len(chunks)
                    })
                    
                    track_metric("documents_processed")
                    
                    # Allow manual indexing (for MVP simplicity)
                    if st.button("Index Document Manually"):
                        with st.spinner("Indexing document..."):
                            try:
                                # For MVP simplicity, just index the first chunk
                                # In a real app, you'd want to index all chunks with proper metadata
                                metadata = {
                                    "source": uploaded_file.name,
                                    "chunk": 1,
                                    "total_chunks": len(chunks),
                                    "content_preview": chunks[0][:100],
                                    "question": f"What information is in {uploaded_file.name}?",
                                    "answer": chunks[0]
                                }
                                
                                search_engine.index_document(
                                    chunks[0],
                                    metadata
                                )
                                
                                st.success("Document indexed successfully!")
                                
                                log_activity("document_indexed", {
                                    "filename": uploaded_file.name,
                                    "chunk_count": 1
                                })
                                
                                track_metric("chunks_indexed")
                            except Exception as e:
                                st.error(f"Error indexing document: {str(e)}")
                                logger.error(f"Error indexing document: {e}")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                logger.error(f"Error processing document: {e}")
        
        # Question input section
        st.header("Ask a Question")
        query = st.text_input("Enter your RFP question")
        
        col_search, col_clear = st.columns([1, 1])
        
        with col_search:
            search_clicked = st.button("Search", type="primary")
        
        with col_clear:
            clear_clicked = st.button("Clear Results")
        
        if clear_clicked:
            if 'search_results' in st.session_state:
                del st.session_state.search_results
                del st.session_state.formatted_results
                del st.session_state.query
        
        if search_clicked and query:
            with st.spinner("Searching..."):
                try:
                    start_time = time.time()
                    
                    # Perform search with LLM enhancement if enabled
                    if use_llm:
                        # Use the combined search and answer method
                        results = search_engine.search_and_answer(
                            query, 
                            limit=search_limit
                        )
                    else:
                        # Use just vector search
                        search_results = search_engine.search(
                            query, 
                            limit=search_limit
                        )
                        results = {
                            "query": query,
                            "search_results": search_results,
                            "generated_answer": search_results[0]["payload"]["answer"] if search_results else "No results found.",
                            "confidence": search_results[0]["score"] if search_results else 0.0
                        }
                    
                    search_time = time.time() - start_time
                    
                    # Format results
                    formatted_results = answer_formatter.format_results(results, query)
                    display_data = answer_formatter.format_for_display(formatted_results)
                    
                    # Save to session state
                    st.session_state.search_results = results
                    st.session_state.formatted_results = formatted_results
                    st.session_state.display_data = display_data
                    st.session_state.query = query
                    st.session_state.search_time = search_time
                    st.session_state.used_llm = use_llm
                    
                    # Log activity
                    log_activity("search", {
                        "query": query,
                        "results_count": len(results.get("search_results", [])),
                        "search_time": search_time,
                        "used_llm": use_llm
                    })
                    
                    track_metric("searches_performed")
                    
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    logger.error(f"Error during search: {e}")
    
    # Right column for results
    with col2:
        st.header("Results")
        
        if 'search_results' in st.session_state:
            st.subheader(f"Results for: {st.session_state.query}")
            search_results = st.session_state.search_results.get("search_results", [])
            st.text(f"Found {len(search_results)} results in {st.session_state.search_time:.2f} seconds")
            
            if st.session_state.used_llm:
                st.text("Enhanced with LLM-generated answer")
            
            # Display confidence
            st.markdown(f"**Confidence**: {st.session_state.display_data['confidence_indicator']} ({st.session_state.display_data['confidence_pct']})")
            
            # Display the answer
            st.markdown("### Answer")
            st.markdown(st.session_state.formatted_results["answer"])
            
            # Feedback buttons
            col_helpful, col_not_helpful = st.columns([1, 1])
            
            with col_helpful:
                if st.button("👍 Helpful"):
                    log_activity("feedback", {
                        "query": st.session_state.query,
                        "feedback": "helpful",
                        "confidence": st.session_state.formatted_results["confidence"],
                        "used_llm": st.session_state.used_llm
                    })
                    
                    track_metric("positive_feedback")
                    st.success("Thank you for your feedback!")
            
            with col_not_helpful:
                if st.button("👎 Not Helpful"):
                    log_activity("feedback", {
                        "query": st.session_state.query,
                        "feedback": "not_helpful",
                        "confidence": st.session_state.formatted_results["confidence"],
                        "used_llm": st.session_state.used_llm
                    })
                    
                    track_metric("negative_feedback")
                    st.error("Thank you for your feedback!")
            
            # Sources
            st.markdown("### Sources")
            for i, source in enumerate(st.session_state.display_data["sources"]):
                with st.expander(f"Source {i+1} - Score: {source['score']:.2f}"):
                    st.markdown(f"**Question:** {source['metadata']['question']}")
                    st.markdown(f"**Answer:** {source['text']}")
                    
                    if source['metadata']['date']:
                        st.text(f"Date: {source['metadata']['date']}")
                    
                    if source['metadata']['answer_type']:
                        st.text(f"Type: {source['metadata']['answer_type']}")

# Add a function for JSON upload and processing
def process_json_data():
    st.header("Process JSON Data")
    
    uploaded_json = st.file_uploader(
        "Upload JSON file with RFP Q&A data",
        type=["json"],
        key="json_uploader"
    )
    
    if uploaded_json:
        try:
            # Read the JSON data
            data = json.load(uploaded_json)
            
            # Preview the data
            st.subheader("Data Preview")
            
            if "documents" in data:
                st.text(f"Found {len(data['documents'])} documents in JSON")
                
                # Show first document as example
                if data["documents"]:
                    st.json(data["documents"][0])
                
                # Process button
                if st.button("Process and Index JSON Data"):
                    with st.spinner("Processing JSON data..."):
                        search_engine, _, _ = initialize_components()
                        
                        # Process similar to your original script
                        documents = []
                        for idx, item in enumerate(data['documents']):
                            combined_text = f"Question: {item['question']} Answer: {item['answer']}"
                            
                            # Generate embedding
                            embedding = search_engine.get_text_embedding(combined_text)
                            
                            metadata = {
                                "question": item['question'],
                                "answer": item['answer'],
                                "answer_type": item.get("answer_type", ""),
                                "summary": item.get("summary", ""),
                                "date": item.get("date", "")
                            }
                            
                            documents.append({
                                "id": idx,
                                "embedding": embedding,
                                "metadata": metadata
                            })
                        
                        # Create collection if needed
                        try:
                            search_engine.create_collection()
                        except Exception as e:
                            st.warning(f"Collection creation warning: {e}")
                        
                        # Index documents
                        search_engine.bulk_index_documents(documents)
                        
                        st.success(f"Successfully processed and indexed {len(documents)} documents!")
                        
                        # Log activity
                        log_activity("json_processing", {
                            "filename": uploaded_json.name,
                            "documents_count": len(documents)
                        })
                        
                        track_metric("json_documents_indexed", len(documents))
            else:
                st.error("Invalid JSON format. Expected a 'documents' array.")
                
        except Exception as e:
            st.error(f"Error processing JSON: {str(e)}")
            logger.error(f"Error processing JSON: {e}")

# Create a navigation menu
def show_navigation():
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Search & Upload", "JSON Data Processing", "Metrics"]
    )
    
    return page

def show_metrics_page():
    st.header("Usage Metrics")
    
    # Check for log file
    try:
        activities = []
        with open("user_activity.log", "r") as f:
            for line in f:
                try:
                    activities.append(json.loads(line))
                except:
                    pass
        
        # Display metrics
        st.subheader("Activity Summary")
        
        if activities:
            # Count by activity type
            activity_types = {}
            for activity in activities:
                activity_type = activity.get("type", "unknown")
                if activity_type not in activity_types:
                    activity_types[activity_type] = 0
                activity_types[activity_type] += 1
            
            # Create DataFrame
            df = pd.DataFrame([
                {"Activity": activity, "Count": count}
                for activity, count in activity_types.items()
            ])
            
            # Display chart
            st.bar_chart(df.set_index("Activity"))
            
            # Display recent activities
            st.subheader("Recent Activities")
            for activity in activities[-10:]:
                st.text(f"{activity['timestamp']} - {activity['type']}")
                
            # Download logs button
            log_data = "\n".join([json.dumps(a) for a in activities])
            st.download_button(
                "Download Activity Log",
                log_data,
                "activity_log.json",
                "application/json"
            )
        else:
            st.info("No activity data available yet")
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        logger.error(f"Error loading metrics: {e}")

def app():
    # Initialize page config
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="📋",
        layout="wide"
    )
    
    # Show navigation and get selected page
    page = show_navigation()
    
    # Display the selected page
    if page == "Search & Upload":
        main()
    elif page == "JSON Data Processing":
        process_json_data()
    elif page == "Metrics":
        show_metrics_page()

if __name__ == "__main__":
    app()