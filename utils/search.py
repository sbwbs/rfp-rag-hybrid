from typing import List, Dict, Any
import logging
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchText

# Set up logging
logger = logging.getLogger(__name__)

class SearchEngine:
    """Handles vector search operations using Qdrant."""
    
    def __init__(self, 
                 openai_api_key: str,
                 qdrant_url: str, 
                 qdrant_api_key: str,
                 collection_name: str = "qa_collection2",
                 embedding_model: str = "text-embedding-3-small",
                 vector_size: int = 512,
                 llm_model: str = "gpt-4o"):
        """Initialize Qdrant client and OpenAI client."""
        logger.info("Initializing SearchEngine")
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_size = vector_size
        self.llm_model = llm_model
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        logger.info(f"SearchEngine initialized with collection: {collection_name}")
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate text embeddings using the specified OpenAI model."""
        try:
            text = text.replace("\n", " ")  # Clean the input text
            response = self.openai_client.embeddings.create(
                input=[text], 
                model=self.embedding_model, 
                dimensions=self.vector_size
            )
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def create_collection(self) -> None:
        """Create a Qdrant collection with the specified vector size."""
        try:
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection {self.collection_name} created/recreated successfully")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def index_document(self, text: str, metadata: Dict[str, Any], doc_id: int = None) -> int:
        """Index document text into Qdrant."""
        try:
            embedding = self.get_text_embedding(text)
            
            if doc_id is None:
                # Get the next available ID
                try:
                    collection_info = self.qdrant_client.get_collection(self.collection_name)
                    doc_id = collection_info.points_count + 1
                except:
                    doc_id = 1
            
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload=metadata
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Document indexed with ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise
    
    def bulk_index_documents(self, documents: List[Dict]) -> None:
        """Insert multiple documents into Qdrant in bulk."""
        try:
            points = [
                PointStruct(
                    id=doc['id'],
                    vector=doc['embedding'],
                    payload=doc['metadata']
                )
                for doc in documents
            ]
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Bulk indexed {len(points)} documents")
        except Exception as e:
            logger.error(f"Error bulk indexing documents: {e}")
            raise
    
    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for relevant document chunks."""
        try:
            query_embedding = self.get_text_embedding(query)
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Convert Qdrant results to a more usable format
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            logger.info(f"Search completed with {len(results)} results for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def generate_answer_from_results(self, query: str, search_results: List[Dict]) -> Dict:
        """
        Generate an improved answer using OpenAI LLM based on search results.
        """
        try:
            if not search_results:
                return {
                    "generated_answer": "No relevant information found.",
                    "confidence": 0.0
                }
            
            # Format context from search results
            context = ""
            for i, result in enumerate(search_results):
                context += f"Source {i+1}:\n"
                if "question" in result["payload"]:
                    context += f"Question: {result['payload']['question']}\n"
                if "answer" in result["payload"]:
                    context += f"Answer: {result['payload']['answer']}\n"
                context += f"Relevance Score: {result['score']:.2f}\n\n"
            
            # Create a prompt for the LLM
            prompt = f"""
                        You are an RFP (Request for Proposal) answering assistant. 
                        Use the provided context from a semantic search to answer the user's question accurately.
                        Only use information from the provided context. If the context doesn't contain enough 
                        information to answer the question fully, acknowledge the limitations in your response.

                        User Question: {query}

                        Context from search results:
                        {context}

                        Instructions:
                        1. Answer the question directly and precisely
                        2. If multiple sources provide relevant information, synthesize them
                        3. If information is incomplete, acknowledge it in your response
                        4. Include any relevant dates, certifications, or specific details mentioned in the context
                        5. Do not make up information that isn't explicitly stated in the context

                        Your answer:
                        """
            
            # Generate answer using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an RFP assistant that provides clear, accurate answers based on the retrieved information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=2000
            )
            
            # Calculate a confidence score based on search result relevance and multiple sources
            top_relevance = search_results[0]["score"] if search_results else 0
            source_diversity = min(len(search_results) / 3, 1.0)  # Normalize to max of 1.0
            confidence = (top_relevance * 0.7) + (source_diversity * 0.3)  # Weighted combination
            
            generated_answer = response.choices[0].message.content
            
            logger.info(f"Generated answer for query: {query[:50]}...")
            
            return {
                "generated_answer": generated_answer,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "generated_answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0
            }
    
    def search_and_answer(self, query: str, limit: int = 3) -> Dict:
        """
        Perform semantic search and generate an answer using OpenAI.
        """
        # Get search results
        search_results = self.search(query, limit)
        
        # Generate answer from results
        answer_data = self.generate_answer_from_results(query, search_results)
        
        # Combine everything into a single response
        return {
            "query": query,
            "search_results": search_results,
            "generated_answer": answer_data["generated_answer"],
            "confidence": answer_data["confidence"]
        }