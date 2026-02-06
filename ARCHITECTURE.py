"""
INFINITE DEBATE SYSTEM - ARCHITECTURE DOCUMENTATION

This document provides a comprehensive overview of the multi-agent conversational
system architecture, design principles, and usage patterns.

═══════════════════════════════════════════════════════════════════════════════
SYSTEM OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

The Infinite Debate System is a production-ready, multi-agent conversational
framework designed to facilitate autonomous debates between AI agents. The system
is built on several key architectural principles:

1. STATELESS AGENTS
   - Agents have no memory beyond what's observable in ConversationState
   - Each agent reads the current state and emits exactly one utterance per turn
   - Agents cannot directly communicate or condition on each other

2. CENTRALIZED STATE MANAGEMENT
   - ConversationState is the sole communication interface between agents
   - All conversation history is immutable once appended
   - No retrospective editing allowed

3. EXPLICIT PROTOCOL CONTROL
   - Turn-taking is governed by code-level A2A (Agent-to-Agent) protocol
   - Agents cannot modify or reason about the protocol
   - Protocol ensures symmetric interaction patterns

4. OBSERVABILITY AND EXTENSIBILITY
   - LangGraph integration provides structured workflow observation
   - Event callbacks enable monitoring and metrics collection
   - Modular design supports easy extension and customization

═══════════════════════════════════════════════════════════════════════════════
CORE COMPONENTS
═══════════════════════════════════════════════════════════════════════════════

1. CONVERSATION STATE (conversation_state.py)
───────────────────────────────────────────────────────────────────────────────
The central state object that serves as the sole observation interface for agents.

Key Classes:
  - ConversationState: Main state container
  - Turn: Individual utterance record
  - ProtocolParameters: Configuration for protocol behavior
  - ConversationMetrics: Derived statistics

Key Methods:
  - append_utterance(): Add utterance to transcript (immutable)
  - get_context_for_agent(): Provide agent's observation window
  - compute_metrics(): Calculate conversation statistics
  - should_terminate(): Check termination conditions

Design Principles:
  - Immutability: Once appended, utterances cannot be modified
  - Symmetry: Both agents have equal access through the same interface
  - Transparency: Full conversation history is always available

2. AGENTS (agents.py)
───────────────────────────────────────────────────────────────────────────────
Autonomous agents that generate debate responses based on ConversationState.

Key Classes:
  - Agent: Abstract base class defining agent interface
  - DebateAgent: Concrete implementation using Groq LLM
  - AgentFactory: Factory for creating agent pairs

Key Methods:
  - generate_utterance(): Core method that produces one response
  - _construct_prompt(): Build prompt from conversation context
  - _build_conversation_history(): Format transcript for LLM

Agent Characteristics:
  - Stateless: No persistent memory between turns
  - Context-driven: All decisions based on ConversationState
  - Single-utterance: Produces exactly one response per invocation

3. PROTOCOL (protocol.py)
───────────────────────────────────────────────────────────────────────────────
The A2A protocol controller that orchestrates turn-taking at the system level.

Key Classes:
  - A2AProtocol: Main protocol controller
  - ProtocolEvent: Event types for monitoring
  - ProtocolBuilder: Factory for protocol creation

Key Methods:
  - run_turn(): Execute a single atomic turn
  - run_conversation(): Run complete conversation to termination
  - _execute_turn(): Atomic unit of protocol execution

Protocol Flow:
  1. Check termination conditions
  2. Get next agent based on ConversationState
  3. Agent observes state and generates utterance
  4. Utterance appended verbatim to ConversationState
  5. Update next speaker
  6. Check termination conditions again

4. WORKFLOW (workflow.py)
───────────────────────────────────────────────────────────────────────────────
LangGraph-based workflow providing structured, observable debate execution.

Key Classes:
  - DebateWorkflow: LangGraph workflow implementation
  - DebateGraphState: State schema for workflow
  - WorkflowBuilder: Factory for workflow creation

Workflow Nodes:
  - agent_turn: Executes agent utterance generation
  - check_termination: Evaluates termination conditions

Workflow Benefits:
  - Structured state management
  - Visual workflow representation
  - Stream-based execution with observability
  - Integration with LangGraph ecosystem

5. UTILITIES (utils.py)
───────────────────────────────────────────────────────────────────────────────
Supporting utilities for logging, persistence, and formatting.

Key Classes:
  - ConversationPersistence: Save/load conversations
  - MetricsTracker: Track and analyze events
  - DebateFormatter: Format output for display

Key Functions:
  - setup_logging(): Configure system logging
  - validate_api_key(): Validate Groq API key
  - format_duration(): Human-readable time formatting

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL PATTERNS
═══════════════════════════════════════════════════════════════════════════════

1. OPEN DYNAMICAL PROCESS
───────────────────────────────────────────────────────────────────────────────
The system implements an open dynamical process where:

- State evolves through discrete, atomic turns
- Each turn is a complete observation-action cycle
- No hidden state or back-channels between agents
- All information flows through ConversationState

This ensures:
  ✓ Reproducibility (same initial state → same trajectory)
  ✓ Auditability (complete conversation history)
  ✓ Debuggability (clear turn-by-turn execution)

2. SYMMETRIC AGENT INTERACTION
───────────────────────────────────────────────────────────────────────────────
Both agents are completely symmetric:

- Same observation interface (ConversationState.get_context_for_agent)
- Same action space (single utterance per turn)
- Same protocol constraints (cannot modify turn-taking)
- Equal informational access (can see full transcript)

This prevents:
  ✗ Asymmetric advantages
  ✗ Hidden communication channels
  ✗ Protocol manipulation

3. IMMUTABLE TRANSCRIPT
───────────────────────────────────────────────────────────────────────────────
Once an utterance is appended:

- It cannot be edited or deleted
- It becomes part of the permanent record
- All agents observe the same history
- Metrics are computed from immutable data

This ensures:
  ✓ Conversation integrity
  ✓ Accurate metrics
  ✓ Trust in the system

═══════════════════════════════════════════════════════════════════════════════
USAGE PATTERNS
═══════════════════════════════════════════════════════════════════════════════

BASIC USAGE
───────────────────────────────────────────────────────────────────────────────
```python
from main import InfiniteDebateSystem

# Initialize system
system = InfiniteDebateSystem(api_key="your_groq_api_key")

# Run debate
final_state = system.run_debate_with_protocol(
    topic="Should AI be regulated?",
    max_turns=10
)

# Display results
system.display_conversation(final_state)
```

ADVANCED USAGE - MANUAL SETUP
───────────────────────────────────────────────────────────────────────────────
```python
from agents import AgentFactory
from conversation_state import ConversationState, ProtocolParameters
from protocol import ProtocolBuilder

# Create agents
agent_1, agent_2 = AgentFactory.create_debate_pair(
    api_key=api_key,
    topic=topic,
    agent_1_stance="Position A",
    agent_2_stance="Position B"
)

# Configure protocol
params = ProtocolParameters(max_turns=20, temperature=0.9)
state = ConversationState(topic=topic, protocol_params=params)

# Create and run protocol
protocol = ProtocolBuilder.create_custom_protocol(agent_1, agent_2, state)
final_state = protocol.run_conversation()
```

LANGGRAPH WORKFLOW USAGE
───────────────────────────────────────────────────────────────────────────────
```python
from workflow import WorkflowBuilder

# Create workflow
workflow = WorkflowBuilder.create_standard_workflow(
    agent_1=agent_1,
    agent_2=agent_2,
    conversation_state=state
)

# Run with streaming
final_state = workflow.run(max_turns=10)
```

═══════════════════════════════════════════════════════════════════════════════
EXTENSION POINTS
═══════════════════════════════════════════════════════════════════════════════

1. CUSTOM AGENTS
───────────────────────────────────────────────────────────────────────────────
Extend the Agent base class to create custom agent behaviors:

```python
from agents import Agent

class CustomAgent(Agent):
    def generate_utterance(self, context, temperature, model, max_tokens):
        # Custom generation logic
        pass
```

2. CUSTOM TERMINATION CONDITIONS
───────────────────────────────────────────────────────────────────────────────
Modify ConversationState.should_terminate() to implement custom conditions:

```python
def should_terminate(self):
    # Add custom logic (e.g., sentiment analysis, topic drift detection)
    if self.detect_agreement():
        return True, "consensus_reached"
    return False, None
```

3. CUSTOM METRICS
───────────────────────────────────────────────────────────────────────────────
Extend ConversationMetrics or create custom metric classes:

```python
from conversation_state import ConversationMetrics

class EnhancedMetrics(ConversationMetrics):
    sentiment_scores: List[float] = []
    topic_coherence: float = 0.0
```

4. EVENT MONITORING
───────────────────────────────────────────────────────────────────────────────
Use protocol event callbacks for custom monitoring:

```python
def custom_event_handler(event, data):
    if event == ProtocolEvent.TURN_COMPLETED:
        # Custom logging, metrics, or actions
        analyze_turn_quality(data)

protocol = ProtocolBuilder.create_standard_protocol(
    agent_1, agent_2, topic,
    event_callback=custom_event_handler
)
```

═══════════════════════════════════════════════════════════════════════════════
PERFORMANCE CONSIDERATIONS
═══════════════════════════════════════════════════════════════════════════════

1. CONTEXT WINDOW MANAGEMENT
   - Use get_context_for_agent(context_window=N) to limit history
   - Larger windows = better context, slower generation
   - Recommended: 10-20 turns for most debates

2. STREAMING RESPONSES
   - Groq API streaming is enabled by default
   - Reduces perceived latency
   - Allows for early termination if needed

3. PARALLEL EXECUTION
   - Agents are stateless and can be parallelized
   - Protocol enforces sequential turn-taking
   - Consider parallel transcript analysis/metrics computation

4. PERSISTENCE
   - Save conversations asynchronously
   - Use compression for large transcripts
   - Implement pagination for long conversations

═══════════════════════════════════════════════════════════════════════════════
TESTING AND VALIDATION
═══════════════════════════════════════════════════════════════════════════════

1. UNIT TESTS
   - Test each component in isolation
   - Mock ConversationState for agent testing
   - Validate protocol turn-taking logic

2. INTEGRATION TESTS
   - Test full conversation flows
   - Validate state consistency
   - Check termination conditions

3. LOAD TESTS
   - Test with long conversations (100+ turns)
   - Monitor memory usage
   - Validate persistence under load

4. VALIDATION CHECKS
   - Turn count consistency (agent_1_turns + agent_2_turns = total_turns)
   - Next speaker alternation
   - Timestamp ordering
   - Transcript immutability

═══════════════════════════════════════════════════════════════════════════════
DEPLOYMENT CONSIDERATIONS
═══════════════════════════════════════════════════════════════════════════════

1. API KEY MANAGEMENT
   - Use environment variables (never hardcode)
   - Implement key rotation
   - Monitor API usage and rate limits

2. LOGGING
   - Configure appropriate log levels for production
   - Use structured logging for analysis
   - Implement log rotation

3. MONITORING
   - Track conversation metrics
   - Monitor API latency and errors
   - Set up alerts for failures

4. SCALABILITY
   - Use message queues for debate requests
   - Implement conversation pooling
   - Consider distributed state management

═══════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

Common Issues:

1. "Turn violation" errors
   → Check that protocol is managing turns correctly
   → Verify agents aren't being called out of order

2. Empty utterances
   → Check API key validity
   → Verify model availability
   → Inspect prompt construction

3. Conversation not terminating
   → Verify max_turns is set
   → Check termination conditions
   → Implement timeout mechanisms

4. High latency
   → Reduce context window
   → Lower max_tokens
   → Check API performance

═══════════════════════════════════════════════════════════════════════════════
LICENSE AND ATTRIBUTION
═══════════════════════════════════════════════════════════════════════════════

This codebase is designed for open-source release. When using or modifying:

- Maintain attribution to original architecture
- Document any modifications to core protocol
- Share improvements back to the community

═══════════════════════════════════════════════════════════════════════════════
"""

# This file serves as living documentation and can be imported for reference
__doc__ = __doc__

__version__ = "1.0.0"
__author__ = "Infinite Debate System Contributors"
__architecture__ = "Open Dynamical Multi-Agent System"
