"""
LangGraph Workflow Module

Implements the debate conversation as a LangGraph StateGraph, providing
a structured, observable workflow for the multi-agent debate system.
"""

from typing import Dict, Any, TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
import logging
from conversation_state import ConversationState
from agents import Agent
from protocol import ProtocolEvent

logger = logging.getLogger(__name__)


class DebateGraphState(TypedDict):
    """
    State schema for the LangGraph debate workflow.
    
    This mirrors key information from ConversationState but is maintained
    separately for LangGraph's state management.
    """
    topic: str
    current_turn: int
    next_speaker: Literal["agent_1", "agent_2"]
    is_terminated: bool
    termination_reason: str | None
    last_utterance: str | None
    last_speaker: str | None
    total_turns: int
    conversation_state_snapshot: Dict[str, Any]


class DebateWorkflow:
    """
    LangGraph-based workflow for managing the debate conversation.
    
    This provides a structured, stateful workflow on top of the core
    protocol and conversation state system.
    """
    
    def __init__(
        self,
        agent_1: Agent,
        agent_2: Agent,
        conversation_state: ConversationState
    ):
        """
        Initialize the debate workflow.
        
        Args:
            agent_1: First agent instance
            agent_2: Second agent instance
            conversation_state: Shared conversation state
        """
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.conversation_state = conversation_state
        
        self.agents_map = {
            "agent_1": agent_1,
            "agent_2": agent_2
        }
        
        logger.info("Initialized DebateWorkflow with LangGraph")
    
    def _initialize_state(self) -> DebateGraphState:
        """
        Initialize the LangGraph state from ConversationState.
        
        Returns:
            Initial graph state
        """
        return DebateGraphState(
            topic=self.conversation_state.topic,
            current_turn=0,
            next_speaker="agent_1",
            is_terminated=False,
            termination_reason=None,
            last_utterance=None,
            last_speaker=None,
            total_turns=0,
            conversation_state_snapshot=self.conversation_state.to_dict()
        )
    
    def _check_should_continue(self, state: DebateGraphState) -> str:
        """
        Conditional edge function - determines if conversation should continue.
        
        Args:
            state: Current graph state
        
        Returns:
            "continue" to proceed with next turn, "end" to terminate
        """
        # Check ConversationState for termination
        should_terminate, reason = self.conversation_state.should_terminate()
        
        if should_terminate or state["is_terminated"]:
            logger.info(f"Workflow terminating: {reason or state['termination_reason']}")
            return "end"
        
        return "continue"
    
    def _agent_turn_node(self, state: DebateGraphState) -> DebateGraphState:
        """
        LangGraph node that executes a single agent turn.
        
        Args:
            state: Current graph state
        
        Returns:
            Updated graph state
        """
        next_speaker_id = state["next_speaker"]
        agent = self.agents_map[next_speaker_id]
        
        logger.info(
            f"Executing turn {state['current_turn'] + 1} for {next_speaker_id}"
        )
        
        try:
            # Get context from ConversationState
            context = self.conversation_state.get_context_for_agent(next_speaker_id)
            
            # Generate utterance
            utterance = agent.generate_utterance(
                context=context,
                temperature=self.conversation_state.protocol_params.temperature,
                model=self.conversation_state.protocol_params.model,
                max_tokens=self.conversation_state.protocol_params.max_utterance_length
            )
            
            # Append to ConversationState
            self.conversation_state.append_utterance(next_speaker_id, utterance)
            
            # Update graph state
            new_turn = state["current_turn"] + 1
            new_next_speaker = "agent_2" if next_speaker_id == "agent_1" else "agent_1"
            
            updated_state = DebateGraphState(
                topic=state["topic"],
                current_turn=new_turn,
                next_speaker=new_next_speaker,
                is_terminated=False,
                termination_reason=None,
                last_utterance=utterance,
                last_speaker=next_speaker_id,
                total_turns=new_turn,
                conversation_state_snapshot=self.conversation_state.to_dict()
            )
            
            logger.info(
                f"Turn {new_turn} completed by {next_speaker_id} "
                f"({len(utterance)} chars)"
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(
                f"Error in agent turn for {next_speaker_id}: {e}",
                exc_info=True
            )
            
            # Mark as terminated due to error
            return DebateGraphState(
                topic=state["topic"],
                current_turn=state["current_turn"],
                next_speaker=state["next_speaker"],
                is_terminated=True,
                termination_reason=f"error_{type(e).__name__}",
                last_utterance=state["last_utterance"],
                last_speaker=state["last_speaker"],
                total_turns=state["total_turns"],
                conversation_state_snapshot=self.conversation_state.to_dict()
            )
    
    def _termination_check_node(self, state: DebateGraphState) -> DebateGraphState:
        """
        LangGraph node that checks termination conditions.
        
        Args:
            state: Current graph state
        
        Returns:
            Updated graph state with termination status
        """
        should_terminate, reason = self.conversation_state.should_terminate()
        
        if should_terminate:
            logger.info(f"Termination condition met: {reason}")
            self.conversation_state.terminate(reason)
            
            return DebateGraphState(
                topic=state["topic"],
                current_turn=state["current_turn"],
                next_speaker=state["next_speaker"],
                is_terminated=True,
                termination_reason=reason,
                last_utterance=state["last_utterance"],
                last_speaker=state["last_speaker"],
                total_turns=state["total_turns"],
                conversation_state_snapshot=self.conversation_state.to_dict()
            )
        
        return state
    
    def build_graph(self) -> CompiledStateGraph:
        """
        Build the LangGraph StateGraph for the debate workflow.
        
        Returns:
            Compiled state graph
        """
        # Create state graph
        workflow = StateGraph(DebateGraphState)
        
        # Add nodes
        workflow.add_node("agent_turn", self._agent_turn_node)
        workflow.add_node("check_termination", self._termination_check_node)
        
        # Set entry point
        workflow.set_entry_point("agent_turn")
        
        # Add edges
        workflow.add_edge("agent_turn", "check_termination")
        
        # Add conditional edge for continuation
        workflow.add_conditional_edges(
            "check_termination",
            self._check_should_continue,
            {
                "continue": "agent_turn",
                "end": END
            }
        )
        
        # Compile graph
        compiled_graph = workflow.compile()
        
        logger.info("LangGraph workflow compiled successfully")
        
        return compiled_graph
    
    def run(self, max_turns: int | None = None) -> ConversationState:
        """
        Run the complete debate workflow.
        
        Args:
            max_turns: Maximum number of turns (overrides protocol params)
        
        Returns:
            Final ConversationState
        """
        logger.info("Starting LangGraph debate workflow")
        
        # Override max_turns if provided
        if max_turns is not None:
            self.conversation_state.protocol_params.max_turns = max_turns
        
        # Build graph
        graph = self.build_graph()
        
        # Initialize state
        initial_state = self._initialize_state()
        
        # Run graph
        try:
            final_state = None
            for state in graph.stream(initial_state):
                final_state = state
                
                # Log progress
                if isinstance(state, dict):
                    for node_name, node_state in state.items():
                        if isinstance(node_state, dict):
                            logger.debug(
                                f"Node '{node_name}' completed - "
                                f"Turn: {node_state.get('current_turn', 'N/A')}"
                            )
            
            logger.info(
                f"Workflow completed - Total turns: "
                f"{self.conversation_state.current_turn_number}"
            )
            
        except Exception as e:
            logger.error(f"Error during workflow execution: {e}", exc_info=True)
            self.conversation_state.terminate(f"workflow_error_{type(e).__name__}")
        
        return self.conversation_state


class WorkflowBuilder:
    """
    Builder for creating debate workflows with various configurations.
    """
    
    @staticmethod
    def create_standard_workflow(
        agent_1: Agent,
        agent_2: Agent,
        conversation_state: ConversationState
    ) -> DebateWorkflow:
        """
        Create a standard debate workflow.
        
        Args:
            agent_1: First agent
            agent_2: Second agent
            conversation_state: Conversation state instance
        
        Returns:
            Configured DebateWorkflow
        """
        workflow = DebateWorkflow(
            agent_1=agent_1,
            agent_2=agent_2,
            conversation_state=conversation_state
        )
        
        logger.info("Created standard workflow")
        
        return workflow
