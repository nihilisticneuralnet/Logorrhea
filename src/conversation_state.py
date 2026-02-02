"""
Conversation State Module

This module defines the ConversationState class that serves as the central,
non-agentic communication interface between agents. It persists the full
transcript, protocol parameters, and derived metrics without allowing
retrospective editing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Represents a single turn in the conversation."""
    agent_id: Literal["agent_1", "agent_2"]
    utterance: str
    timestamp: datetime
    turn_number: int
    
    def to_dict(self) -> Dict:
        """Convert turn to dictionary format."""
        return {
            "agent_id": self.agent_id,
            "utterance": self.utterance,
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number
        }


@dataclass
class ProtocolParameters:
    """Protocol parameters that govern the debate."""
    max_turns: Optional[int] = None
    max_utterance_length: int = 2000
    temperature: float = 1.0
    model: str = "llama-3.1-8b-instant"
    turn_timeout_seconds: int = 30
    
    def to_dict(self) -> Dict:
        """Convert parameters to dictionary format."""
        return {
            "max_turns": self.max_turns,
            "max_utterance_length": self.max_utterance_length,
            "temperature": self.temperature,
            "model": self.model,
            "turn_timeout_seconds": self.turn_timeout_seconds
        }


@dataclass
class ConversationMetrics:
    """Derived metrics computed from the conversation state."""
    total_turns: int = 0
    agent_1_turns: int = 0
    agent_2_turns: int = 0
    total_tokens_estimate: int = 0
    conversation_duration_seconds: float = 0.0
    average_utterance_length: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        return {
            "total_turns": self.total_turns,
            "agent_1_turns": self.agent_1_turns,
            "agent_2_turns": self.agent_2_turns,
            "total_tokens_estimate": self.total_tokens_estimate,
            "conversation_duration_seconds": self.conversation_duration_seconds,
            "average_utterance_length": self.average_utterance_length
        }


class ConversationState:
    """
    Central state object for the multi-agent debate system.
    
    This class serves as the sole observation interface for both agents.
    Agents are stateless and can only access conversation history through
    this object. No retrospective editing is allowed - all utterances are
    appended verbatim and immutably to the transcript.
    """
    
    def __init__(
        self,
        topic: str,
        protocol_params: Optional[ProtocolParameters] = None
    ):
        """
        Initialize a new conversation state.
        
        Args:
            topic: The debate topic
            protocol_params: Protocol parameters (uses defaults if None)
        """
        self.topic = topic
        self.protocol_params = protocol_params or ProtocolParameters()
        self.transcript: List[Turn] = []
        self.conversation_start_time = datetime.now()
        self.current_turn_number = 0
        self.next_speaker: Literal["agent_1", "agent_2"] = "agent_1"
        self._is_terminated = False
        self._termination_reason: Optional[str] = None
        
        logger.info(
            f"Initialized ConversationState with topic: '{topic}'"
        )
    
    def append_utterance(
        self,
        agent_id: Literal["agent_1", "agent_2"],
        utterance: str
    ) -> None:
        """
        Append an utterance to the transcript verbatim.
        
        This is the only way to add content to the conversation.
        No editing or deletion is permitted.
        
        Args:
            agent_id: ID of the agent making the utterance
            utterance: The utterance text (appended verbatim)
        
        Raises:
            ValueError: If it's not the agent's turn or conversation is terminated
        """
        if self._is_terminated:
            raise ValueError(
                f"Cannot append utterance: conversation already terminated "
                f"({self._termination_reason})"
            )
        
        if agent_id != self.next_speaker:
            raise ValueError(
                f"Turn violation: {agent_id} attempted to speak but "
                f"next speaker is {self.next_speaker}"
            )
        
        # Create turn object
        self.current_turn_number += 1
        turn = Turn(
            agent_id=agent_id,
            utterance=utterance,
            timestamp=datetime.now(),
            turn_number=self.current_turn_number
        )
        
        # Append verbatim to transcript (immutable operation)
        self.transcript.append(turn)
        
        # Update next speaker
        self.next_speaker = "agent_2" if agent_id == "agent_1" else "agent_1"
        
        logger.info(
            f"Turn {self.current_turn_number}: {agent_id} spoke "
            f"({len(utterance)} chars). Next: {self.next_speaker}"
        )
    
    def get_full_transcript(self) -> List[Turn]:
        """
        Get the complete, immutable transcript.
        
        Returns:
            List of all turns in chronological order
        """
        return self.transcript.copy()
    
    def get_transcript_text(self) -> str:
        """
        Get the transcript as formatted text.
        
        Returns:
            String representation of the full conversation
        """
        lines = [f"TOPIC: {self.topic}\n"]
        for turn in self.transcript:
            agent_label = turn.agent_id.replace("_", " ").title()
            lines.append(f"{agent_label}: {turn.utterance}\n")
        return "\n".join(lines)
    
    def get_context_for_agent(
        self,
        agent_id: Literal["agent_1", "agent_2"],
        context_window: Optional[int] = None
    ) -> Dict:
        """
        Get the observation context for a specific agent.
        
        This is the sole interface through which agents observe the conversation.
        Agents cannot directly access each other or communicate outside this.
        
        Args:
            agent_id: ID of the requesting agent
            context_window: Number of recent turns to include (None = all)
        
        Returns:
            Dictionary containing topic, relevant transcript, and metadata
        """
        # Select transcript window
        if context_window is not None and context_window > 0:
            transcript_window = self.transcript[-context_window:]
        else:
            transcript_window = self.transcript
        
        # Build context
        context = {
            "topic": self.topic,
            "agent_id": agent_id,
            "is_my_turn": self.next_speaker == agent_id,
            "turn_number": self.current_turn_number,
            "transcript": [turn.to_dict() for turn in transcript_window],
            "my_turns_count": sum(
                1 for t in self.transcript if t.agent_id == agent_id
            ),
            "opponent_turns_count": sum(
                1 for t in self.transcript 
                if t.agent_id != agent_id
            )
        }
        
        return context
    
    def compute_metrics(self) -> ConversationMetrics:
        """
        Compute derived metrics from the current state.
        
        Returns:
            ConversationMetrics object with current statistics
        """
        metrics = ConversationMetrics()
        
        metrics.total_turns = len(self.transcript)
        metrics.agent_1_turns = sum(
            1 for t in self.transcript if t.agent_id == "agent_1"
        )
        metrics.agent_2_turns = sum(
            1 for t in self.transcript if t.agent_id == "agent_2"
        )
        
        # Estimate tokens (rough approximation: 1 token ~= 4 chars)
        total_chars = sum(len(t.utterance) for t in self.transcript)
        metrics.total_tokens_estimate = total_chars // 4
        
        # Calculate duration
        if self.transcript:
            duration = (
                self.transcript[-1].timestamp - self.conversation_start_time
            )
            metrics.conversation_duration_seconds = duration.total_seconds()
        
        # Average utterance length
        if self.transcript:
            metrics.average_utterance_length = (
                total_chars / len(self.transcript)
            )
        
        return metrics
    
    def should_terminate(self) -> tuple[bool, Optional[str]]:
        """
        Check if conversation should terminate based on protocol parameters.
        
        Returns:
            Tuple of (should_terminate: bool, reason: Optional[str])
        """
        if self._is_terminated:
            return True, self._termination_reason
        
        # Check max turns
        if (self.protocol_params.max_turns is not None and 
            self.current_turn_number >= self.protocol_params.max_turns):
            return True, f"max_turns_reached ({self.protocol_params.max_turns})"
        
        return False, None
    
    def terminate(self, reason: str) -> None:
        """
        Terminate the conversation.
        
        Args:
            reason: Reason for termination
        """
        if not self._is_terminated:
            self._is_terminated = True
            self._termination_reason = reason
            logger.info(f"Conversation terminated: {reason}")
    
    def is_terminated(self) -> bool:
        """Check if conversation is terminated."""
        return self._is_terminated
    
    def get_termination_reason(self) -> Optional[str]:
        """Get the reason for termination if terminated."""
        return self._termination_reason
    
    def to_dict(self) -> Dict:
        """
        Export full conversation state to dictionary.
        
        Returns:
            Complete state as dictionary
        """
        metrics = self.compute_metrics()
        
        return {
            "topic": self.topic,
            "protocol_params": self.protocol_params.to_dict(),
            "transcript": [turn.to_dict() for turn in self.transcript],
            "metrics": metrics.to_dict(),
            "conversation_start_time": self.conversation_start_time.isoformat(),
            "current_turn_number": self.current_turn_number,
            "next_speaker": self.next_speaker,
            "is_terminated": self._is_terminated,
            "termination_reason": self._termination_reason
        }
