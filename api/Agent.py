import os
import getpass
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Type
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



# Set up logger for tracking tool usage
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global variable to store the vector database instance
therapy_vector_db = None

class RAGSearchInput(BaseModel):
    query: str

class TherapyRAGTool(BaseTool):
    name: str = "therapy_document_search"
    description: str = "Search the therapy document for information about common human problems and solutions. Use this FIRST for any mental health, emotional, or life problem questions."
    args_schema: Type[BaseModel] = RAGSearchInput
    
    def _run(self, query: str) -> str:
        """Search the therapy document for relevant information."""
        global therapy_vector_db
        
        if therapy_vector_db is None:
            return "Therapy document not available. I'll provide general supportive advice."
        
        try:
            # Search for relevant chunks
            relevant_chunks = therapy_vector_db.search_by_text(query, k=3, return_as_text=True)
            
            if not relevant_chunks or len(relevant_chunks) == 0:
                return f"No specific information found in therapy document for: {query}. May need to search for additional resources."
            
            # Format the results
            context = "\n\n".join(relevant_chunks)
            return f"Found relevant information in therapy document:\n\n{context}"
            
        except Exception as e:
            return f"Error searching therapy document: {str(e)}. I'll provide general guidance."

def set_therapy_vector_db(vector_db):
    """Set the global vector database instance."""
    global therapy_vector_db
    therapy_vector_db = vector_db







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



  # inputs = {"messages" : [HumanMessage(content="Who is the current captain of the Winnipeg Jets?")]}

  # async for chunk in simple_agent_graph.astream(inputs, stream_mode="updates"):
  #     for node, values in chunk.items():
  #         print(f"Receiving update from node: '{node}'")
  #         print(values["messages"])
  #         print("\n\n")

  # inputs = {"messages" : [HumanMessage(content="Search Arxiv for the QLoRA paper, then search each of the authors to find out their latest Tweet using Tavily!")]}

  # async for chunk in simple_agent_graph.astream(inputs, stream_mode="updates"):
  #     for node, values in chunk.items():
  #         print(f"Receiving update from node: '{node}'")
  #         if node == "action":
  #           print(f"Tool Used: {values['messages'][0].name}")
  #         print(values["messages"])

  #         print("\n\n")

# def convert_inputs(input_object):
#   return {"messages" : [HumanMessage(content=input_object["question"])]}

# def parse_output(input_state):
#   return input_state["messages"][-1].content

# agent_chain_with_formatting = convert_inputs | simple_agent_graph | parse_output

# agent_chain_with_formatting.invoke({"question" : "What is RAG?"})

# questions = [
#     "What optimizer is used in QLoRA?",
#     "What data type was created in the QLoRA paper?",
#     "What is a Retrieval Augmented Generation system?",
#     "Who authored the QLoRA paper?",
#     "What is the most popular deep learning framework?",
#     "What significant improvements does the LoRA system make?"
# ]

# answers = [
#     {"must_mention" : ["paged", "optimizer"]},
#     {"must_mention" : ["NF4", "NormalFloat"]},
#     {"must_mention" : ["ground", "context"]},
#     {"must_mention" : ["Tim", "Dettmers"]},
#     {"must_mention" : ["PyTorch", "TensorFlow"]},
#     {"must_mention" : ["reduce", "parameters"]},
# ]



# client = Client()

# dataset_name = f"Retrieval Augmented Generation - Evaluation Dataset - {uuid4().hex[0:8]}"

# dataset = client.create_dataset(
#     dataset_name=dataset_name,
#     description="Questions about the QLoRA Paper to Evaluate RAG over the same paper."
# )

# client.create_examples(
#     inputs=[{"question" : q} for q in questions],
#     outputs=answers,
#     dataset_id=dataset.id,
# )



# @run_evaluator
# def must_mention(run, example) -> EvaluationResult:
#     prediction = run.outputs.get("output") or ""
#     required = example.outputs.get("must_mention") or []
#     score = all(phrase in prediction for phrase in required)
#     return EvaluationResult(key="must_mention", score=score)

# experiment_results = client.evaluate(
#     agent_chain_with_formatting,
#     data=dataset_name,
#     evaluators=[must_mention],
#     experiment_prefix=f"Search Pipeline - Evaluation - {uuid4().hex[0:4]}",
#     metadata={"version": "1.0.0"},
# )

# experiment_results

# class AgentState(TypedDict):
#   messages: Annotated[list, add_messages]

# graph_with_helpfulness_check = StateGraph(AgentState)

# graph_with_helpfulness_check.add_node("agent", call_model)
# graph_with_helpfulness_check.add_node("action", tool_node)

# graph_with_helpfulness_check.set_entry_point("agent")



# def tool_call_or_helpful(state):
#   last_message = state["messages"][-1]

#   if last_message.tool_calls:
#     return "action"

#   initial_query = state["messages"][0]
#   final_response = state["messages"][-1]

#   if len(state["messages"]) > 10:
#     return "END"

#   prompt_template = """\
#   Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

#   Initial Query:
#   {initial_query}

#   Final Response:
#   {final_response}"""

#   helpfullness_prompt_template = PromptTemplate.from_template(prompt_template)

#   helpfulness_check_model = ChatOpenAI(model="gpt-4.1-mini")

#   helpfulness_chain = helpfullness_prompt_template | helpfulness_check_model | StrOutputParser()

#   helpfulness_response = helpfulness_chain.invoke({"initial_query" : initial_query.content, "final_response" : final_response.content})

#   if "Y" in helpfulness_response:
#     return "end"
#   else:
#     return "continue"

# graph_with_helpfulness_check.add_conditional_edges(
#     "agent",
#     tool_call_or_helpful,
#     {
#         "continue" : "agent",
#         "action" : "action",
#         "end" : END
#     }
# )

# graph_with_helpfulness_check.add_edge("action", "agent")

# agent_with_helpfulness_check = graph_with_helpfulness_check.compile()

# inputs = {"messages" : [HumanMessage(content="Related to machine learning, what is LoRA? Also, who is Tim Dettmers? Also, what is Attention?")]}

# async for chunk in agent_with_helpfulness_check.astream(inputs, stream_mode="updates"):
#     for node, values in chunk.items():
#         print(f"Receiving update from node: '{node}'")
#         print(values["messages"])
#         print("\n\n")

# patterns = ["prompt engineering", "RAG", "fine-tuning", "LLM-based agents"]

# for pattern in patterns:
#   what_is_string = f"What is {pattern} and when did it break onto the scene??"
#   inputs = {"messages" : [HumanMessage(content=what_is_string)]}
#   messages = agent_with_helpfulness_check.invoke(inputs)
#   print(messages["messages"][-1].content)
#   print("\n\n")