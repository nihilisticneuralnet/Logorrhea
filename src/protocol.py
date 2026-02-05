"""
Protocol Module

Defines the explicit, code-level Agent-to-Agent (A2A) protocol that governs
turn-taking and conversation flow. Agents cannot modify or reason about
this protocol - it operates at the system level.
"""

from typing import Dict, Literal, Optional, Callable
from enum import Enum
import logging
from datetime import datetime
from conversation_state import ConversationState, ProtocolParameters
from agents import Agent
from tts import DebateTTSGenerator, TTSConfig

logger = logging.getLogger(__name__)


class ProtocolEvent(Enum):
    """Events that can occur during protocol execution."""
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    UTTERANCE_GENERATED = "utterance_generated"
    UTTERANCE_APPENDED = "utterance_appended"
    ERROR_OCCURRED = "error_occurred"
    TERMINATION_CHECKED = "termination_checked"
    CONVERSATION_TERMINATED = "conversation_terminated"


class A2AProtocol:
    """
    Agent-to-Agent Protocol Controller.
    
    This class implements the explicit turn-taking mechanism that agents
    cannot modify or reason about. It orchestrates the conversation flow
    by managing agent turns through the ConversationState interface.
    """
    
    def __init__(
        self,
        agent_1: Agent,
        agent_2: Agent,
        conversation_state: ConversationState,
        event_callback: Optional[Callable[[ProtocolEvent, Dict], None]] = None,
        tts_generator: Optional[DebateTTSGenerator] = None
    ):
        """
        Initialize the A2A protocol controller.
        
        Args:
            agent_1: First agent instance
            agent_2: Second agent instance
            conversation_state: Shared conversation state object
            event_callback: Optional callback for protocol events
            tts_generator: Optional TTS generator for audio output
        """
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.conversation_state = conversation_state
        self.event_callback = event_callback
        self.tts_generator = tts_generator
        
        self.agents_map = {
            "agent_1": agent_1,
            "agent_2": agent_2
        }
        
        logger.info("Initialized A2A Protocol Controller")
        if self.tts_generator:
            logger.info("TTS generation enabled")
    
    def _emit_event(
        self,
        event: ProtocolEvent,
        data: Optional[Dict] = None
    ) -> None:
        """
        Emit a protocol event.
        
        Args:
            event: Protocol event type
            data: Optional event data
        """
        if self.event_callback:
            event_data = data or {}
            event_data["event"] = event.value
            event_data["timestamp"] = str(logger)
            try:
                self.event_callback(event, event_data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}", exc_info=True)
    
    def _get_next_agent(self) -> Agent:
        """
        Get the next agent to speak based on ConversationState.
        
        Returns:
            Agent instance for next turn
        """
        next_speaker_id = self.conversation_state.next_speaker
        return self.agents_map[next_speaker_id]
    
    def _execute_turn(self) -> bool:
        """
        Execute a single turn in the conversation.
        
        This is the atomic unit of the protocol: get context, generate
        utterance, append to state.
        
        Returns:
            True if turn completed successfully, False otherwise
        """
        # Get next agent
        agent = self._get_next_agent()
        agent_id = agent.agent_id
        
        self._emit_event(
            ProtocolEvent.TURN_STARTED,
            {"agent_id": agent_id, "turn": self.conversation_state.current_turn_number + 1}
        )
        
        try:
            # Agent observes conversation state (sole observation interface)
            context = self.conversation_state.get_context_for_agent(agent_id)
            
            # Agent generates exactly one utterance
            logger.info(f"Requesting utterance from {agent_id}")
            utterance = agent.generate_utterance(
                context=context,
                temperature=self.conversation_state.protocol_params.temperature,
                model=self.conversation_state.protocol_params.model,
                max_tokens=self.conversation_state.protocol_params.max_utterance_length
            )
            
            self._emit_event(
                ProtocolEvent.UTTERANCE_GENERATED,
                {
                    "agent_id": agent_id,
                    "utterance_length": len(utterance),
                    "utterance_preview": utterance[:100]
                }
            )
            
            # Utterance is appended verbatim to ConversationState
            self.conversation_state.append_utterance(agent_id, utterance)
            
            # Store in Mem0 if available
            if (hasattr(self.conversation_state, 'mem0_manager') and 
                self.conversation_state.mem0_manager):
                try:
                    from conversation_state import Turn
                    turn = Turn(
                        agent_id=agent_id,
                        utterance=utterance,
                        timestamp=datetime.now(),
                        turn_number=self.conversation_state.current_turn_number
                    )
                    self.conversation_state.mem0_manager.store_turn(
                        turn,
                        self.conversation_state.topic
                    )
                    logger.debug(f"Stored turn in Mem0 memory")
                except Exception as e:
                    logger.error(f"Mem0 storage error: {e}")
            
            self._emit_event(
                ProtocolEvent.UTTERANCE_APPENDED,
                {
                    "agent_id": agent_id,
                    "turn_number": self.conversation_state.current_turn_number
                }
            )
            
            # Generate TTS audio in real-time if enabled
            if self.tts_generator and self.tts_generator.config.enable_realtime:
                try:
                    logger.info(f"Generating TTS audio for {agent_id}...")
                    self.tts_generator.add_utterance(
                        text=utterance,
                        agent_id=agent_id,
                        turn_number=self.conversation_state.current_turn_number,
                        add_pause=True
                    )
                except Exception as e:
                    logger.error(f"TTS generation error: {e}", exc_info=True)
            
            self._emit_event(
                ProtocolEvent.TURN_COMPLETED,
                {
                    "agent_id": agent_id,
                    "turn_number": self.conversation_state.current_turn_number
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Error during turn execution for {agent_id}: {e}",
                exc_info=True
            )
            self._emit_event(
                ProtocolEvent.ERROR_OCCURRED,
                {
                    "agent_id": agent_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return False
    
    def _check_termination(self) -> tuple[bool, Optional[str]]:
        """
        Check if conversation should terminate.
        
        Returns:
            Tuple of (should_terminate, reason)
        """
        should_terminate, reason = self.conversation_state.should_terminate()
        
        self._emit_event(
            ProtocolEvent.TERMINATION_CHECKED,
            {
                "should_terminate": should_terminate,
                "reason": reason
            }
        )
        
        return should_terminate, reason
    
    def run_turn(self) -> bool:
        """
        Execute a single turn and check for termination.
        
        Returns:
            True if conversation should continue, False if terminated
        """
        # Check termination before turn
        should_terminate, reason = self._check_termination()
        if should_terminate:
            self.conversation_state.terminate(reason)
            self._emit_event(
                ProtocolEvent.CONVERSATION_TERMINATED,
                {"reason": reason}
            )
            return False
        
        # Execute turn
        success = self._execute_turn()
        
        if not success:
            # Turn failed - terminate conversation
            self.conversation_state.terminate("turn_execution_failed")
            self._emit_event(
                ProtocolEvent.CONVERSATION_TERMINATED,
                {"reason": "turn_execution_failed"}
            )
            return False
        
        # Check termination after turn
        should_terminate, reason = self._check_termination()
        if should_terminate:
            self.conversation_state.terminate(reason)
            self._emit_event(
                ProtocolEvent.CONVERSATION_TERMINATED,
                {"reason": reason}
            )
            return False
        
        return True
    
    def run_conversation(
        self,
        max_turns: Optional[int] = None
    ) -> ConversationState:
        """
        Run the complete conversation until termination.
        
        Args:
            max_turns: Maximum number of turns (overrides protocol params if set)
        
        Returns:
            Final ConversationState
        """
        logger.info("Starting conversation execution")
        
        # Override max_turns if provided
        if max_turns is not None:
            self.conversation_state.protocol_params.max_turns = max_turns
        
        # Main protocol loop
        while not self.conversation_state.is_terminated():
            should_continue = self.run_turn()
            
            if not should_continue:
                break
        
        logger.info(
            f"Conversation completed: {self.conversation_state.current_turn_number} turns, "
            f"reason: {self.conversation_state.get_termination_reason()}"
        )
        
        return self.conversation_state


class ProtocolBuilder:
    """
    Builder for creating A2A protocol instances with various configurations.
    """
    
    @staticmethod
    def create_standard_protocol(
        agent_1: Agent,
        agent_2: Agent,
        topic: str,
        max_turns: Optional[int] = None,
        temperature: float = 1.0,
        event_callback: Optional[Callable] = None,
        enable_tts: bool = False,
        tts_config: Optional[TTSConfig] = None
    ) -> A2AProtocol:
        """
        Create a standard debate protocol.
        
        Args:
            agent_1: First agent
            agent_2: Second agent
            topic: Debate topic
            max_turns: Maximum turns (None for infinite)
            temperature: LLM temperature
            event_callback: Optional event callback
            enable_tts: Whether to enable TTS generation
            tts_config: Optional TTS configuration
        
        Returns:
            Configured A2AProtocol instance
        """
        protocol_params = ProtocolParameters(
            max_turns=max_turns,
            temperature=temperature
        )
        
        conversation_state = ConversationState(
            topic=topic,
            protocol_params=protocol_params
        )
        
        # Create TTS generator if enabled
        tts_generator = None
        if enable_tts:
            tts_generator = DebateTTSGenerator(config=tts_config)
            logger.info("TTS generation enabled for protocol")
        
        protocol = A2AProtocol(
            agent_1=agent_1,
            agent_2=agent_2,
            conversation_state=conversation_state,
            event_callback=event_callback,
            tts_generator=tts_generator
        )
        
        logger.info(
            f"Created standard protocol for topic: '{topic}' "
            f"(max_turns: {max_turns})"
        )
        
        return protocol
    
    @staticmethod
    def create_custom_protocol(
        agent_1: Agent,
        agent_2: Agent,
        conversation_state: ConversationState,
        event_callback: Optional[Callable] = None,
        tts_generator: Optional[DebateTTSGenerator] = None
    ) -> A2AProtocol:
        """
        Create a protocol with custom conversation state.
        
        Args:
            agent_1: First agent
            agent_2: Second agent
            conversation_state: Pre-configured conversation state
            event_callback: Optional event callback
            tts_generator: Optional TTS generator
        
        Returns:
            Configured A2AProtocol instance
        """
        protocol = A2AProtocol(
            agent_1=agent_1,
            agent_2=agent_2,
            conversation_state=conversation_state,
            event_callback=event_callback,
            tts_generator=tts_generator
        )
        
        logger.info("Created custom protocol")
        
        return protocol
