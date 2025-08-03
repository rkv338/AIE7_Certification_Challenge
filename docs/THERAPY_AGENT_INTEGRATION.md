# Therapy Agent Integration - RAG + Tavily

## Overview

Your therapy support application now uses an intelligent Agent that combines:
1. **RAG First**: Searches your therapy document for relevant guidance on common human problems
2. **Tavily Fallback**: Conducts web research for topics not covered in the document
3. **Empathetic Responses**: Provides caring yet constructive therapeutic guidance

## How It Works

### Agent Flow
1. **User Question** → Therapy Agent
2. **Agent Decision**: Always tries `therapy_document_search` tool first
3. **RAG Search**: Looks for relevant information in the uploaded therapy document
4. **Evaluation**: If document has insufficient information for the user's situation
5. **Tavily Research**: Searches web for additional resources and current information
6. **Response**: Combines findings into empathetic, actionable guidance

### Key Components

#### TherapyRAGTool
```python
class TherapyRAGTool(BaseTool):
    name: str = "therapy_document_search"
    description: str = "Search the therapy document for information about common human problems and solutions. Use this FIRST for any mental health, emotional, or life problem questions."
```

#### System Instructions
The agent is programmed with therapy-specific guidelines:
- Always validate feelings before offering solutions
- Provide empathetic yet constructive responses
- Encourage professional help when appropriate
- Cite sources clearly (document vs. web research)

## API Integration

### Updated `/api/chat-rag` Endpoint

**Before**: Direct RAG → LLM response
**Now**: User Question → Agent (RAG + Tavily) → Therapeutic Response

```python
@app.post("/api/chat-rag")
async def chat_rag(request: RAGChatRequest):
    # Set up agent with therapy vector DB
    set_therapy_vector_db(vector_db)
    agent_graph = create_agent_graph()
    
    # Agent processes with RAG first, Tavily if needed
    # Returns streaming empathetic response
```

## Example Scenarios

### 1. Document-Based Response
**User**: "I'm feeling anxious about work presentations"
- **Agent Action**: Uses `therapy_document_search`
- **Result**: Finds relevant anxiety management techniques in therapy document
- **Response**: "I understand how challenging work presentations can feel. Based on therapeutic guidance, here are some strategies that can help..."

### 2. Web Research Fallback
**User**: "I'm struggling with remote work isolation since 2024"
- **Agent Action**: Tries `therapy_document_search` first
- **Document Result**: General isolation advice found
- **Agent Decision**: Current context (2024, remote work) needs additional research
- **Tavily Research**: Searches for latest remote work mental health strategies
- **Response**: Combines document guidance with current best practices

### 3. Combined Approach
**User**: "How do I deal with social media addiction?"
- **Agent Action**: Searches therapy document for addiction/digital wellness guidance
- **Additional Research**: Uses Tavily for latest digital wellness trends
- **Response**: Therapeutic framework + current tools and strategies

## Response Characteristics

### Therapeutic Approach
- **Validation**: "I hear that you're going through a difficult time..."
- **Empathy**: "It's completely understandable to feel this way..."
- **Constructive Guidance**: "Here are some strategies that might help..."
- **Encouragement**: "Remember, seeking support is a sign of strength..."

### Source Attribution
- **Document-based**: "Based on therapeutic guidance..." 
- **Web research**: "Additional resources suggest..."
- **Combined**: "Therapeutic approaches combined with current research indicate..."

## Environment Setup

Ensure you have these environment variables:

```bash
# Required
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# Optional (LangChain tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AIE7-Therapy-Agent
```

## Benefits

1. **Comprehensive Coverage**: Document knowledge + current web information
2. **Intelligent Routing**: RAG first, research when needed
3. **Therapeutic Quality**: Empathetic, professional, constructive responses
4. **Contextual Awareness**: Understands when additional research is needed
5. **Streaming Responses**: Real-time, engaging user experience

## Frontend Usage

The existing frontend works unchanged - just point to `/api/chat-rag`:

```javascript
const response = await fetch('http://localhost:8000/api/chat-rag', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_message: "I'm struggling with anxiety",
    pdf_id: 'therapy-doc',
    api_key: process.env.NEXT_PUBLIC_OPENAI_API_KEY,
  }),
});
```

The agent will automatically:
- Search the therapy document first
- Use web research if needed
- Provide caring, constructive guidance
- Stream the response for better UX

Your therapy support system is now more intelligent and comprehensive! 🌟