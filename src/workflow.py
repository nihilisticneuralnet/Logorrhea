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
    def __init__(
        self,
        agent_1: Agent,
        agent_2: Agent,
        conversation_state: ConversationState
    ):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.conversation_state = conversation_state
        
        self.agents_map = {
            "agent_1": agent_1,
            "agent_2": agent_2
        }
        
        logger.info("Initialized DebateWorkflow with LangGraph")
    
    def _initialize_state(self) -> DebateGraphState:
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
        should_terminate, reason = self.conversation_state.should_terminate()
        
        if should_terminate or state["is_terminated"]:
            logger.info(f"Workflow terminating: {reason or state['termination_reason']}")
            return "end"
        
        return "continue"
    
    def _agent_turn_node(self, state: DebateGraphState) -> DebateGraphState:
        next_speaker_id = state["next_speaker"]
        agent = self.agents_map[next_speaker_id]
        
        logger.info(
            f"Executing turn {state['current_turn'] + 1} for {next_speaker_id}"
        )
        
        try:
            context = self.conversation_state.get_context_for_agent(next_speaker_id)
            
            utterance = agent.generate_utterance(
                context=context,
                temperature=self.conversation_state.protocol_params.temperature,
                model=self.conversation_state.protocol_params.model,
                max_tokens=self.conversation_state.protocol_params.max_utterance_length
            )
            
            self.conversation_state.append_utterance(next_speaker_id, utterance)
            
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
        workflow = StateGraph(DebateGraphState)
        
        workflow.add_node("agent_turn", self._agent_turn_node)
        workflow.add_node("check_termination", self._termination_check_node)
        
        workflow.set_entry_point("agent_turn")
        workflow.add_edge("agent_turn", "check_termination")
        workflow.add_conditional_edges(
            "check_termination",
            self._check_should_continue,
            {
                "continue": "agent_turn",
                "end": END
            }
        )
        
        compiled_graph = workflow.compile()
        
        logger.info("LangGraph workflow compiled successfully")
        
        return compiled_graph
    
    def run(self, max_turns: int | None = None) -> ConversationState:
        logger.info("Starting LangGraph debate workflow")
        
        if max_turns is not None:
            self.conversation_state.protocol_params.max_turns = max_turns
        
        graph = self.build_graph()
        
        initial_state = self._initialize_state()
        
        try:
            final_state = None
            for state in graph.stream(initial_state):
                final_state = state
                
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
    @staticmethod
    def create_standard_workflow(
        agent_1: Agent,
        agent_2: Agent,
        conversation_state: ConversationState
    ) -> DebateWorkflow:
        workflow = DebateWorkflow(
            agent_1=agent_1,
            agent_2=agent_2,
            conversation_state=conversation_state
        )
        
        logger.info("Created standard workflow")
        
        return workflow
