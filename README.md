# RFP Q&A System

A Streamlit-based application for semantic search of RFP (Request for Proposal) documents, providing question-answering capabilities using vector search with Qdrant.

## Features

- **Document Upload & Processing**: Support for PDF, DOCX, XLSX, and TXT files
- **Vector Search**: Semantic search using OpenAI embeddings and Qdrant vector database  
- **JSON Data Import**: Bulk import of Q&A pairs from JSON format
- **User-Friendly Interface**: Simple, Google Translate-like UI for easy interaction
- **Usage Metrics**: Track and visualize system usage and performance

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key
- Qdrant account (cloud or self-hosted)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/sbwbs/rfp-rag-hybrid.git
   cd rfp-rag-hybrid
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

### Running the Application

Start the Streamlit application:

```
streamlit run app.py
```

## Usage

### Document Upload

1. Navigate to the "Search & Upload" page
2. Upload your RFP document (PDF, DOCX, XLSX, or TXT)
3. View the extracted text preview
4. Click "Index Document Manually" to store in Qdrant

### JSON Data Processing

1. Navigate to the "JSON Data Processing" page
2. Upload a JSON file with the following structure:
   ```json
   {
     "documents": [
       {
         "question": "What certifications does the company have?",
         "answer": "We are ISO 9001 and ISO 27001 certified.",
         "answer_type": "certification",
         "date": "2023-01-01"
       },
       ...
     ]
   }
   ```
3. Click "Process and Index JSON Data" to embed and store in Qdrant

### Asking Questions

1. Enter your question in the input field
2. Click "Search" to retrieve relevant answers
3. View the results, confidence level, and sources
4. Provide feedback with the thumbs up/down buttons

## Project Structure

```
rfp-rag-hybrid/
├── utils
│   ├── answer_formatter.py  # Formats search results into readable answers
│   ├── document_processor.py  # Handles document text extraction
│   └── search.py  # Handles vector search operations
├── app.py  # Main Streamlit application
├── config.py  # Configuration settings
└── requirements.txt  # Project dependencies
```

## Future Enhancements

- Implement hybrid search (vector + keyword)
- Add support for more document types
- Improve document chunking strategies
- Add user authentication
- Implement more advanced answer generation using LLMs

## Evaluation

The system tracks the following metrics:

- Documents processed
- Searches performed
- User feedback (helpful/not helpful)
- Search latency
- Confidence scores

View these metrics on the "Metrics" page.

## License

This project is licensed under the MIT License - see the LICENSE file for details.