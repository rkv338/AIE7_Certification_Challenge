import os
import getpass
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Type, List
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langsmith.evaluation import EvaluationResult, run_evaluator
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import Client
from uuid import uuid4
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langchain_core.tools import BaseTool
from pydantic import BaseModel

# LangChain Ensemble Retriever imports
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

# LangChain Contextual Compression Retriever imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor



# Set up logger for tracking tool usage
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global variables for ensemble retrieval
therapy_vector_db = None
therapy_ensemble_retriever = None
therapy_compression_retriever = None
therapy_documents = []

class RAGSearchInput(BaseModel):
    query: str

class TherapyRAGTool(BaseTool):
    name: str = "therapy_document_search"
    description: str = "Search the therapy document for information about common human problems and solutions. Use this FIRST for any mental health, emotional, or life problem questions. Uses contextual compression with ensemble retrieval (BM25 + semantic search) for precise, relevant therapeutic guidance."
    args_schema: Type[BaseModel] = RAGSearchInput
    
    def _run(self, query: str) -> str:
        """Search the therapy document for relevant information using contextual compression."""
        global therapy_compression_retriever, therapy_ensemble_retriever, therapy_vector_db
        
        # Use compression retriever if available (wraps ensemble)
        if therapy_compression_retriever is not None:
            try:
                relevant_docs = therapy_compression_retriever.get_relevant_documents(query)
                
                if relevant_docs and len(relevant_docs) > 0:
                    # Extract content from compressed documents
                    relevant_chunks = [doc.page_content for doc in relevant_docs[:3]]
                    context = "\n\n".join(relevant_chunks)
                    return f"Found relevant information in therapy document (compressed ensemble search):\n\n{context}"
                
            except Exception as e:
                logger.error(f"Compression retriever error: {e}")
                # Fall through to ensemble retriever
        
        # Fallback to ensemble retriever if compression fails or isn't available
        if therapy_ensemble_retriever is not None:
            try:
                relevant_docs = therapy_ensemble_retriever.get_relevant_documents(query)
                
                if relevant_docs and len(relevant_docs) > 0:
                    # Extract content from retrieved documents
                    relevant_chunks = [doc.page_content for doc in relevant_docs[:3]]
                    context = "\n\n".join(relevant_chunks)
                    return f"Found relevant information in therapy document (ensemble search):\n\n{context}"
                
            except Exception as e:
                logger.error(f"Ensemble retriever error: {e}")
                # Fall through to basic vector search
        
        # Fallback to basic vector search
        if therapy_vector_db is not None:
            try:
                relevant_chunks = therapy_vector_db.search_by_text(query, k=3, return_as_text=True)
                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)
                    return f"Found relevant information in therapy document (basic search):\n\n{context}"
            except Exception as e:
                logger.error(f"Vector search error: {e}")
        
        # No retrieval method available
        return "Therapy document not available. I'll provide general supportive advice."

def set_therapy_vector_db(vector_db, documents_list=None):
    """Set up the global vector database, ensemble retriever, and contextual compression retriever."""
    global therapy_vector_db, therapy_ensemble_retriever, therapy_compression_retriever, therapy_documents
    
    therapy_vector_db = vector_db
    
    try:
        # If we have the original documents list, set up ensemble retriever
        if documents_list is not None:
            therapy_documents = documents_list
            
            # Create Document objects for LangChain
            langchain_docs = [Document(page_content=doc) for doc in documents_list]
            
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(langchain_docs)
            bm25_retriever.k = 3
            
            # Create a custom vector store retriever wrapper
            class CustomVectorStoreRetriever:
                def __init__(self, vector_db):
                    self.vector_db = vector_db
                
                def get_relevant_documents(self, query: str) -> List[Document]:
                    """Get relevant documents using our vector database."""
                    try:
                        # Get results from our vector database
                        results = self.vector_db.search_by_text(query, k=3, return_as_text=True)
                        # Convert to Document objects
                        return [Document(page_content=result) for result in results]
                    except Exception as e:
                        logger.error(f"Vector retriever error: {e}")
                        return []
            
            vector_retriever = CustomVectorStoreRetriever(vector_db)
            
            # Create ensemble retriever with weighted combination
            # 40% BM25 (keyword matching) + 60% semantic (vector similarity)
            therapy_ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6],
                k=3
            )
            
            logger.info("✅ Ensemble retriever successfully created with BM25 + Vector Search")
            
            # Create contextual compression retriever wrapping the ensemble retriever
            try:
                # Create LLM for compression
                compression_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                
                # Create compressor that extracts relevant parts
                compressor = LLMChainExtractor.from_llm(compression_llm)
                
                # Wrap ensemble retriever with contextual compression
                therapy_compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=therapy_ensemble_retriever
                )
                
                logger.info("✅ Contextual compression retriever successfully created wrapping ensemble retriever")
                
            except Exception as compression_error:
                logger.error(f"❌ Failed to create compression retriever: {compression_error}")
                logger.info("🔄 Using ensemble retriever without compression")
                therapy_compression_retriever = None
            
        else:
            logger.warning("⚠️ No documents list provided - using basic vector search only")
            therapy_ensemble_retriever = None
            therapy_compression_retriever = None
            
    except Exception as e:
        logger.error(f"❌ Failed to create ensemble retriever: {e}")
        logger.info("🔄 Falling back to basic vector search")
        therapy_ensemble_retriever = None
        therapy_compression_retriever = None







class AgentState(TypedDict):
  messages: Annotated[list, add_messages]



def call_model(state):
  messages = state["messages"]
  
  # Add system message for therapy context if it's the first human message
  if len(messages) == 1 and messages[0].type == "human":
    therapy_system_message = SystemMessage(content="""You are a caring and empathetic AI therapist assistant. Your role is to help people with their problems in a constructive yet compassionate manner.

IMPORTANT INSTRUCTIONS:
1. ALWAYS use the 'therapy_document_search' tool FIRST for any mental health, emotional, or life problem questions
2. Only use 'tavily_search_results_json' if the therapy document doesn't contain relevant information for the user's specific situation
3. Provide responses that are:
   - Empathetic and understanding
   - Constructive with actionable advice
   - Supportive yet encouraging personal growth
   - Professional but warm in tone

4. When using therapy document information, cite it as "based on therapeutic guidance"
5. When using web research, clearly indicate these are additional resources
6. Always validate the person's feelings before offering solutions
7. Encourage professional help when appropriate

Remember: You're here to listen, support, and provide helpful guidance for life's challenges.""")
    
    messages = [therapy_system_message] + messages
  
  response = model.invoke(messages)
  
  # Log tool usage after the model response
  if hasattr(response, 'tool_calls') and response.tool_calls:
    for tool_call in response.tool_calls:
      tool_name = tool_call.get('name', 'unknown')
      if 'tavily' in tool_name.lower():
        logger.info(f"🌐 TAVILY RESEARCH: Agent decided to use Tavily search for web research")
        # Try to get the query being searched
        try:
          if hasattr(tool_call, 'args') and isinstance(tool_call.args, dict):
            query = tool_call.args.get('query', 'unknown query')
            logger.info(f"🔍 TAVILY QUERY: '{query}'")
        except:
          logger.info("🔍 TAVILY QUERY: Could not extract search query")
      elif 'therapy_document_search' in tool_name:
        logger.info(f"🔍 RAG SEARCH: Agent using therapy document search first")
  
  return {"messages" : [response]}

def should_continue(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  return END

# Global model variable
model = None

def create_agent_graph():
  global model
  
  # Create tools - RAG first, then Tavily
  therapy_rag_tool = TherapyRAGTool()
  tavily_tool = TavilySearchResults(max_results=3)  # Reduced for more focused results

  tool_belt = [
      therapy_rag_tool,  # Try therapy document first
      tavily_tool        # Fallback to web search
  ]

  # Create model with appropriate temperature for therapy responses
  model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)  # Slightly creative for empathy
  model = model.bind_tools(tool_belt)
  tool_node = ToolNode(tool_belt)


  uncompiled_graph = StateGraph(AgentState)

  uncompiled_graph.add_node("agent", call_model)
  uncompiled_graph.add_node("action", tool_node)

  uncompiled_graph.set_entry_point("agent")



  uncompiled_graph.add_conditional_edges(
      "agent",
      should_continue
  )

  uncompiled_graph.add_edge("action", "agent")

  simple_agent_graph = uncompiled_graph.compile()

  return simple_agent_graph