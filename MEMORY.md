
## Three-Tier Memory Architecture

### 1. **Full Transcript** (Persistent Storage)
- Complete, unedited conversation history
- Stored in database/disk (not in LLM context)
- Used for: analysis, replay, metrics, audit trail
- **Size**: Unlimited (grows forever)

### 2. **Rolling Summary** (Compressed Context)
- LLM-generated summary of older conversation segments
- Updated every N turns (e.g., every 50 messages)
- Provides thematic continuity without full history
- **Size**: Fixed (~500-1000 tokens)

### 3. **Recent Messages** (Working Context)
- Last 20-50 messages verbatim
- Immediate conversational context for coherence
- Most critical for natural turn-to-turn flow
- **Size**: Fixed (~3000-5000 tokens)

## How It Works

```
Turn 1-50:     [Full messages in context]
Turn 51-100:   [Summary of 1-50] + [Full messages 51-100]
Turn 101-150:  [Summary of 1-100] + [Full messages 101-150]
...
Turn 10,000:   [Rolling summary] + [Last 50 messages]
```

**Each agent sees**:
```
System: "Debate topic: [topic]"
Summary: "So far, Agent1 argued X, Agent2 countered with Y..."
Recent: [Last 50 messages verbatim]
```

**What agents DON'T see**: The full 10,000 message history (stored separately)

## Implementation Stack

- **LangGraph**: State management, checkpointing, orchestration
- **LangChain**: Message trimming, token counting utilities
- **tiktoken**: Accurate token budget tracking
- **Custom**: Summarization logic, context windowing policy

## Key Design Decisions

✅ **Use existing tools for**: State persistence, token counting, message formatting  
✅ **Build custom logic for**: Summarization strategy, context window policy  

**Why not build from scratch?**  
LangGraph handles edge cases (concurrent state updates, checkpointing, error recovery) that would take weeks to implement correctly.

**Why custom summarization?**  
Domain-specific compression (debate arguments, positions, rebuttals) is more effective than generic summarization.

## Context Management Rules

1. **Trigger summarization**: Every 50 turns
2. **Compress range**: Messages from 50-100 turns ago
3. **Keep recent**: Last 50 messages always in full
4. **Summary prompt**: "Extract key arguments, positions, and unresolved points"
5. **Token budget**: 8000 tokens total per agent view (summary 1K + messages 5K + system 2K)

## ConversationState Schema

```python
{
    # Ground truth (persisted)
    "full_transcript": [...],      # All messages ever
    "protocol_params": {...},      # Turn rules, timing
    
    # Agent view (computed dynamically)
    "working_context": [...],      # Last N messages
    "context_summary": "...",      # Compressed history
    
    # Metadata
    "turn_count": int,
    "current_speaker": str,
    "topic": str
}
```

## Trade-offs

**What you gain**: Infinite scalability, bounded costs  
**What you lose**: Agents may "forget" specific details from 500 turns ago  
**Acceptable because**: Debate coherence relies on recent arguments + thematic continuity, not verbatim recall of turn 347
