"""
Agent Module

Defines autonomous, stateless agents that interact exclusively through
ConversationState. Agents observe the conversation state and emit exactly
one utterance per turn.
"""

from typing import Dict, Literal, Optional
from abc import ABC, abstractmethod
import logging
from groq import Groq

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Abstract base class for debate agents.
    
    Agents are stateless beyond what's observable in ConversationState.
    They read the current state and emit exactly one utterance per turn.
    """
    
    def __init__(
        self,
        agent_id: Literal["agent_1", "agent_2"],
        api_key: str,
        stance: Optional[str] = None,
        personality: Optional[str] = None
    ):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for this agent
            api_key: Groq API key
            stance: Optional stance/position for the agent
            personality: Optional personality description
        """
        self.agent_id = agent_id
        self.client = Groq(api_key=api_key)
        self.stance = stance
        self.personality = personality
        
        logger.info(
            f"Initialized {agent_id} with stance: {stance}, "
            f"personality: {personality}"
        )
    
    @abstractmethod
    def generate_utterance(
        self,
        context: Dict,
        temperature: float,
        model: str,
        max_tokens: int
    ) -> str:
        """
        Generate a single utterance based on conversation context.
        
        This is the core agent behavior - reading ConversationState context
        and producing exactly one response.
        
        Args:
            context: Context dictionary from ConversationState
            temperature: Sampling temperature
            model: Model identifier
            max_tokens: Maximum tokens to generate
        
        Returns:
            Single utterance string
        """
        pass


class DebateAgent(Agent):
    """
    Concrete implementation of a debate agent using Groq LLM.
    
    This agent constructs prompts from ConversationState context and
    generates debate responses through the Groq API.
    """
    
    def __init__(
        self,
        agent_id: Literal["agent_1", "agent_2"],
        api_key: str,
        stance: Optional[str] = None,
        personality: Optional[str] = None,
        system_prompt_template: Optional[str] = None
    ):
        """
        Initialize a debate agent.
        
        Args:
            agent_id: Unique identifier
            api_key: Groq API key
            stance: Agent's stance on debate topics
            personality: Agent's personality traits
            system_prompt_template: Custom system prompt template
        """
        super().__init__(agent_id, api_key, stance, personality)
        self.system_prompt_template = (
            system_prompt_template or self._default_system_prompt()
        )
    
    def _default_system_prompt(self) -> str:
        """
        Create default system prompt for the agent.
        
        Returns:
            System prompt string
        """
        base_prompt = (
            "You are a debate participant engaging in a rigorous, "
            "intellectual discussion. Your goal is to present strong arguments, "
            "respond to your opponent's points, and advance the conversation "
            "with logical reasoning and evidence-based claims.\n\n"
        )
        
        if self.stance:
            base_prompt += f"Your stance: {self.stance}\n\n"
        
        if self.personality:
            base_prompt += f"Your personality: {self.personality}\n\n"
        
        base_prompt += (
            "Guidelines:\n"
            "- Make clear, well-reasoned arguments\n"
            "- Engage directly with your opponent's points\n"
            "- Use logical reasoning and examples\n"
            "- Be respectful but firm in your position\n"
            "- Keep responses focused and coherent\n"
            "- Avoid repetition of previous arguments\n"
            "- Build upon the conversation naturally\n"
        )
        
        return base_prompt
    
    def _build_conversation_history(self, context: Dict) -> str:
        """
        Build conversation history string from context.
        
        Args:
            context: Context dictionary from ConversationState
        
        Returns:
            Formatted conversation history
        """
        history_lines = [f"DEBATE TOPIC: {context['topic']}\n"]
        
        for turn in context['transcript']:
            agent_label = turn['agent_id'].replace('_', ' ').title()
            history_lines.append(f"{agent_label}: {turn['utterance']}")
        
        return "\n\n".join(history_lines)
    
    def _construct_prompt(self, context: Dict) -> str:
        """
        Construct the full prompt for the LLM.
        
        Args:
            context: Context dictionary from ConversationState
        
        Returns:
            Complete prompt string
        """
        if not context['transcript']:
            # First turn - introduce the topic
            prompt = (
                f"You are beginning a debate on the following topic:\n\n"
                f"{context['topic']}\n\n"
                f"Provide your opening statement. Present your position clearly "
                f"and set the stage for a thoughtful discussion."
            )
        else:
            # Subsequent turns - respond to conversation
            history = self._build_conversation_history(context)
            prompt = (
                f"{history}\n\n"
                f"It is now your turn. Respond to the previous argument and "
                f"advance your position. Be direct, logical, and engaging."
            )
        
        return prompt
    
    def generate_utterance(
        self,
        context: Dict,
        temperature: float,
        model: str,
        max_tokens: int
    ) -> str:
        """
        Generate a single utterance using the Groq API.
        
        Args:
            context: Context from ConversationState
            temperature: Sampling temperature
            model: Model identifier
            max_tokens: Maximum completion tokens
        
        Returns:
            Generated utterance
        """
        logger.info(
            f"{self.agent_id} generating utterance "
            f"(turn {context['turn_number'] + 1})"
        )
        
        # Verify it's this agent's turn
        if not context['is_my_turn']:
            raise ValueError(
                f"{self.agent_id} attempted to generate utterance "
                f"but it's not their turn"
            )
        
        # Construct prompt from conversation state
        user_prompt = self._construct_prompt(context)
        
        # Call Groq API
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt_template
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=True,
                stop=None
            )
            
            # Collect streamed response
            utterance_parts = []
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    utterance_parts.append(chunk.choices[0].delta.content)
            
            utterance = "".join(utterance_parts).strip()
            
            if not utterance:
                logger.warning(
                    f"{self.agent_id} generated empty utterance, "
                    f"using fallback"
                )
                utterance = "I need more time to formulate my response."
            
            logger.info(
                f"{self.agent_id} generated utterance "
                f"({len(utterance)} chars)"
            )
            
            return utterance
            
        except Exception as e:
            logger.error(
                f"Error generating utterance for {self.agent_id}: {e}",
                exc_info=True
            )
            # Return error fallback instead of raising
            return (
                f"I apologize, I'm experiencing technical difficulties "
                f"and cannot respond properly at this moment."
            )


class AgentFactory:
    """
    Factory for creating debate agents with predefined configurations.
    """
    
    @staticmethod
    def create_debate_pair(
        api_key: str,
        topic: str,
        agent_1_stance: Optional[str] = None,
        agent_2_stance: Optional[str] = None,
        agent_1_personality: Optional[str] = None,
        agent_2_personality: Optional[str] = None
    ) -> tuple[DebateAgent, DebateAgent]:
        """
        Create a pair of debate agents with opposing or complementary stances.
        
        Args:
            api_key: Groq API key
            topic: Debate topic (used to infer stances if not provided)
            agent_1_stance: Explicit stance for agent 1
            agent_2_stance: Explicit stance for agent 2
            agent_1_personality: Personality for agent 1
            agent_2_personality: Personality for agent 2
        
        Returns:
            Tuple of (agent_1, agent_2)
        """
        # Use provided stances or create generic opposing stances
        stance_1 = agent_1_stance or "Support the affirmative position"
        stance_2 = agent_2_stance or "Support the negative position"
        
        personality_1 = agent_1_personality or (
            "Analytical and evidence-focused, favoring logical arguments "
            "and empirical data"
        )
        personality_2 = agent_2_personality or (
            "Philosophical and principle-based, emphasizing ethical "
            "considerations and thought experiments"
        )
        
        agent_1 = DebateAgent(
            agent_id="agent_1",
            api_key=api_key,
            stance=stance_1,
            personality=personality_1
        )
        
        agent_2 = DebateAgent(
            agent_id="agent_2",
            api_key=api_key,
            stance=stance_2,
            personality=personality_2
        )
        
        logger.info(
            f"Created agent pair for topic: '{topic}'"
        )
        
        return agent_1, agent_2
    
    @staticmethod
    def create_specialized_agents(
        api_key: str,
        agent_1_config: Dict,
        agent_2_config: Dict
    ) -> tuple[DebateAgent, DebateAgent]:
        """
        Create agents with fully custom configurations.
        
        Args:
            api_key: Groq API key
            agent_1_config: Configuration dict for agent 1
            agent_2_config: Configuration dict for agent 2
        
        Returns:
            Tuple of (agent_1, agent_2)
        """
        agent_1 = DebateAgent(
            agent_id="agent_1",
            api_key=api_key,
            **agent_1_config
        )
        
        agent_2 = DebateAgent(
            agent_id="agent_2",
            api_key=api_key,
            **agent_2_config
        )
        
        return agent_1, agent_2
