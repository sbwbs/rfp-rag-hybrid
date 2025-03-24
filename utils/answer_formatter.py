from typing import List, Dict
import logging
import re
import markdown

# Set up logging
logger = logging.getLogger(__name__)

class AnswerFormatter:
    """Formats search results into readable answers."""
    
    def format_results(self, results: Dict, query: str) -> Dict:
        """Format search results and generated answer into a readable response."""
        try:
            logger.info(f"Formatting results for query: {query}")
            
            search_results = results.get("search_results", [])
            generated_answer = results.get("generated_answer", "")
            confidence = results.get("confidence", 0.0)
            
            if not search_results and not generated_answer:
                logger.warning("No results to format")
                return {
                    "answer": "No relevant information found for this query.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Format sources information
            sources = []
            for result in search_results:
                source = {
                    "text": result["payload"].get("answer", ""),
                    "score": result["score"],
                    "metadata": {
                        "question": result["payload"].get("question", ""),
                        "answer_type": result["payload"].get("answer_type", ""),
                        "date": result["payload"].get("date", "")
                    }
                }
                sources.append(source)
            
            # Convert markdown to HTML for better display
            # This step isn't necessary for Streamlit as it natively supports markdown
            # But included here for completeness
            try:
                html_answer = markdown.markdown(generated_answer)
            except:
                html_answer = generated_answer
            
            formatted_response = {
                "answer": generated_answer,
                "html_answer": html_answer,
                "sources": sources,
                "confidence": confidence
            }
            
            logger.info("Successfully formatted search results")
            return formatted_response
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return {
                "answer": f"Error formatting results: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def highlight_relevant_parts(self, text: str, query: str) -> str:
        """Highlight relevant parts of the text based on the query."""
        try:
            logger.debug(f"Highlighting relevant parts for query: {query}")
            
            # Extract key terms from the query (words with 4+ characters)
            key_terms = [term.lower() for term in re.findall(r'\b\w{4,}\b', query)]
            
            # Simple highlighting for MVP
            highlighted_text = text
            for term in key_terms:
                # Case-insensitive replacement with bold markdown
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted_text = pattern.sub(f"**{term}**", highlighted_text)
            
            return highlighted_text
        except Exception as e:
            logger.error(f"Error highlighting text: {e}")
            return text
    
    def add_confidence_indicators(self, confidence: float) -> str:
        """Add visual confidence indicators based on the confidence score."""
        try:
            if confidence >= 0.8:
                return "🟢 High confidence"
            elif confidence >= 0.5:
                return "🟡 Medium confidence"
            else:
                return "🔴 Low confidence"
        except Exception as e:
            logger.error(f"Error adding confidence indicators: {e}")
            return "⚪ Unknown confidence"
    
    def format_for_display(self, formatted_results: Dict) -> Dict:
        """Prepare results for display in the UI."""
        try:
            display_data = formatted_results.copy()
            
            # Add confidence indicator
            display_data["confidence_indicator"] = self.add_confidence_indicators(
                formatted_results["confidence"]
            )
            
            # Format confidence as percentage
            display_data["confidence_pct"] = f"{int(formatted_results['confidence'] * 100)}%"
            
            return display_data
        except Exception as e:
            logger.error(f"Error formatting for display: {e}")
            return formatted_results