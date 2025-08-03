# RAG + Tavily Agent Integration

## Overview

Your AI Agent now has access to both your vector database (RAG) and Tavily web search. The agent intelligently chooses which tool to use based on the question and automatically falls back to web search when the document doesn't contain sufficient information.

## How It Works

1. **RAG First**: The agent always tries the `rag_search` tool first to look for information in your uploaded PDF
2. **Smart Fallback**: If the document search doesn't provide relevant information, the agent automatically uses Tavily search
3. **Intelligent Routing**: The agent is instructed to prefer document sources but use web search for:
   - Current/recent information not likely to be in the document
   - Questions where document search returns no relevant results
   - Topics that require more comprehensive or updated information

## Key Features

### Custom RAG Tool
- **Tool Name**: `rag_search`
- **Description**: "Search the uploaded PDF document for relevant information. Use this first before searching the web."
- **Returns**: Relevant document chunks or "No relevant information found"

### Enhanced Agent Instructions
The agent has system-level instructions that guide it to:
- Always try RAG search first
- Only use Tavily when document search is insufficient
- Clearly cite sources (document vs. web)

## Usage Example

```python
from api.Agent import create_agent_graph
from langchain_core.messages import HumanMessage

# Create the enhanced agent
graph = create_agent_graph()

# Ask a question
inputs = {"messages": [HumanMessage(content="What is anxiety therapy?")]}

# The agent will:
# 1. First search your PDF document
# 2. If insufficient, search the web with Tavily
# 3. Provide a comprehensive answer citing sources

async for chunk in graph.astream(inputs, stream_mode="updates"):
    for node, values in chunk.items():
        if node == "agent":
            print(f"Agent Response: {values['messages'][-1].content}")
        elif node == "action":
            print(f"Using tool: {values['messages'][-1].name}")
```

## Integration with Your App

### Option 1: Direct Integration
You can directly use the agent graph in your FastAPI app:

```python
from api.Agent import create_agent_graph

@app.post("/api/chat-agent")
async def chat_with_agent(request: ChatRequest):
    graph = create_agent_graph()
    inputs = {"messages": [HumanMessage(content=request.user_message)]}
    
    # Stream response from agent
    async def generate():
        async for chunk in graph.astream(inputs, stream_mode="updates"):
            for node, values in chunk.items():
                if node == "agent" and not values['messages'][-1].tool_calls:
                    yield values['messages'][-1].content
    
    return StreamingResponse(generate(), media_type="text/plain")
```

### Option 2: Update Existing RAG Endpoint
Modify your existing RAG endpoint to use the agent for more intelligent responses.

## Environment Variables Required

Make sure you have these set:

```bash
# Required
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# Optional (for LangChain tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AIE7-Enhanced-RAG
```

## Expected Behavior

### Questions About Document Content
**User**: "What is cognitive behavioral therapy?"
- Agent uses `rag_search` first
- Finds relevant information in therapy document
- Responds with document-based information
- Cites document as source

### Questions Requiring Current Information
**User**: "What are the latest AI developments in 2024?"
- Agent tries `rag_search` first
- Finds no relevant information (document likely doesn't have 2024 info)
- Automatically uses `tavily_search_results_json`
- Provides current web-based information
- Cites web sources

### Comprehensive Questions
**User**: "How to deal with anxiety?"
- Agent searches document first
- If document has some info but seems incomplete
- May use both document and web sources
- Combines information intelligently

## Testing

To test the agent, uncomment the test function in `Agent.py`:

```python
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
```

This will run sample questions that demonstrate both RAG and Tavily usage.