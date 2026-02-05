"""
Main Application Module

Entry point for running the infinite debate system. Demonstrates usage
of all components and provides a complete example.
"""

import logging
from typing import Optional
from agents import AgentFactory, DebateAgent
from conversation_state import ConversationState, ProtocolParameters
from protocol import A2AProtocol, ProtocolBuilder, ProtocolEvent
from workflow import DebateWorkflow, WorkflowBuilder
from tts import DebateTTSGenerator, TTSConfig
from memory import DebateMemoryManager
from utils import (
    setup_logging,
    ConversationPersistence,
    MetricsTracker,
    validate_api_key,
    get_environment_variable,
    DebateFormatter
)

logger = logging.getLogger(__name__)


class InfiniteDebateSystem:
    """
    Main system class for orchestrating infinite debates between agents.
    """
    
    def __init__(
        self,
        api_key: str,
        storage_dir: str = "./conversations",
        log_level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize the debate system.
        
        Args:
            api_key: Groq API key
            storage_dir: Directory for saving conversations
            log_level: Logging level
            log_file: Optional log file path
        """
        # Setup logging
        setup_logging(log_level=log_level, log_file=log_file)
        
        # Validate API key
        if not validate_api_key(api_key):
            raise ValueError("Invalid API key format")
        
        self.api_key = api_key
        self.persistence = ConversationPersistence(storage_dir)
        self.metrics_tracker = MetricsTracker()
        
        logger.info("Initialized InfiniteDebateSystem")
    
    def _protocol_event_handler(
        self,
        event: ProtocolEvent,
        data: dict
    ) -> None:
        """
        Handle protocol events for logging and metrics.
        
        Args:
            event: Protocol event type
            data: Event data
        """
        self.metrics_tracker.log_event(event.value, data)
        
        # Log important events
        if event == ProtocolEvent.TURN_COMPLETED:
            logger.info(
                f"Turn {data.get('turn_number')} completed by "
                f"{data.get('agent_id')}"
            )
        elif event == ProtocolEvent.CONVERSATION_TERMINATED:
            logger.info(f"Conversation terminated: {data.get('reason')}")
    
    def run_debate_with_protocol(
        self,
        topic: str,
        max_turns: Optional[int] = 10,
        agent_1_stance: Optional[str] = None,
        agent_2_stance: Optional[str] = None,
        temperature: float = 1.0,
        save_results: bool = True,
        enable_tts: bool = False,
        tts_config: Optional[TTSConfig] = None,
        enable_memory: bool = False,
        memory_user_id_prefix: Optional[str] = None
    ) -> ConversationState:
        """
        Run a debate using the A2A Protocol approach.
        
        Args:
            topic: Debate topic
            max_turns: Maximum number of turns (None for infinite)
            agent_1_stance: Stance for agent 1
            agent_2_stance: Stance for agent 2
            temperature: LLM temperature
            save_results: Whether to save results to disk
            enable_tts: Whether to generate TTS audio
            tts_config: Optional TTS configuration
            enable_memory: Whether to use mem0 semantic memory
            memory_user_id_prefix: Optional prefix for memory user IDs
        
        Returns:
            Final ConversationState
        """
        logger.info(f"Starting debate with Protocol approach - Topic: '{topic}'")
        
        # Create memory manager if enabled
        memory_manager = None
        if enable_memory:
            try:
                memory_manager = DebateMemoryManager(
                    user_id_prefix=memory_user_id_prefix or "debate"
                )
                logger.info("✓ mem0 memory manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize memory manager: {e}", exc_info=True)
                logger.warning("Continuing without memory")
                enable_memory = False
        
        # Create agents with memory support
        agent_1 = DebateAgent(
            agent_id="agent_1",
            api_key=self.api_key,
            stance=agent_1_stance or "Support the affirmative position",
            personality="Analytical and evidence-focused, favoring logical arguments and empirical data",
            # memory_manager=memory_manager,
            # enable_memory=enable_memory
        )
        
        agent_2 = DebateAgent(
            agent_id="agent_2",
            api_key=self.api_key,
            stance=agent_2_stance or "Support the negative position",
            personality="Philosophical and principle-based, emphasizing ethical considerations and thought experiments",
            # memory_manager=memory_manager,
            # enable_memory=enable_memory
        )
        
        if enable_memory:
            logger.info("✓ Agents created with mem0 memory enabled")
        
        # Create protocol with TTS
        protocol = ProtocolBuilder.create_standard_protocol(
            agent_1=agent_1,
            agent_2=agent_2,
            topic=topic,
            max_turns=max_turns,
            temperature=temperature,
            event_callback=self._protocol_event_handler,
            enable_tts=enable_tts,
            tts_config=tts_config
        )
        
        # Run conversation
        final_state = protocol.run_conversation()
        
        # Save TTS audio if enabled
        if enable_tts and protocol.tts_generator:
            try:
                audio_path = protocol.tts_generator.save_debate_audio(topic=topic)
                logger.info(f"✓ Saved debate audio: {audio_path}")
                
                # Log TTS statistics
                stats = protocol.tts_generator.get_statistics()
                logger.info(
                    f"TTS Stats: {stats['total_duration']:.2f}s total, "
                    f"{stats['agent_1_turns']} + {stats['agent_2_turns']} turns"
                )
            except Exception as e:
                logger.error(f"Failed to save TTS audio: {e}", exc_info=True)
        
        # Log memory stats if enabled
        if enable_memory and memory_manager:
            try:
                stats_1 = memory_manager.get_memory_statistics("agent_1")
                stats_2 = memory_manager.get_memory_statistics("agent_2")
                logger.info(
                    f"Memory Stats: Agent 1: {stats_1['total_memories']} memories, "
                    f"Agent 2: {stats_2['total_memories']} memories"
                )
            except Exception as e:
                logger.error(f"Failed to get memory stats: {e}")
        
        # Save results if requested
        if save_results:
            self.persistence.save_conversation(final_state)
            self.persistence.save_transcript(final_state)
        
        logger.info(
            f"Debate completed - {final_state.current_turn_number} turns"
        )
        
        return final_state
    
    def run_debate_with_workflow(
        self,
        topic: str,
        max_turns: Optional[int] = 10,
        agent_1_stance: Optional[str] = None,
        agent_2_stance: Optional[str] = None,
        temperature: float = 1.0,
        save_results: bool = True
    ) -> ConversationState:
        """
        Run a debate using the LangGraph Workflow approach.
        
        Args:
            topic: Debate topic
            max_turns: Maximum number of turns (None for infinite)
            agent_1_stance: Stance for agent 1
            agent_2_stance: Stance for agent 2
            temperature: LLM temperature
            save_results: Whether to save results to disk
        
        Returns:
            Final ConversationState
        """
        logger.info(f"Starting debate with Workflow approach - Topic: '{topic}'")
        
        # Create agents
        agent_1, agent_2 = AgentFactory.create_debate_pair(
            api_key=self.api_key,
            topic=topic,
            agent_1_stance=agent_1_stance,
            agent_2_stance=agent_2_stance
        )
        
        # Create conversation state
        protocol_params = ProtocolParameters(
            max_turns=max_turns,
            temperature=temperature
        )
        conversation_state = ConversationState(
            topic=topic,
            protocol_params=protocol_params
        )
        
        # Create workflow
        workflow = WorkflowBuilder.create_standard_workflow(
            agent_1=agent_1,
            agent_2=agent_2,
            conversation_state=conversation_state
        )
        
        # Run workflow
        final_state = workflow.run(max_turns=max_turns)
        
        # Save results if requested
        if save_results:
            self.persistence.save_conversation(final_state)
            self.persistence.save_transcript(final_state)
        
        logger.info(
            f"Debate completed - {final_state.current_turn_number} turns"
        )
        
        return final_state
    
    def display_conversation(
        self,
        conversation_state: ConversationState,
        format_type: str = "console"
    ) -> None:
        """
        Display a conversation in the specified format.
        
        Args:
            conversation_state: Conversation to display
            format_type: Format type ('console' or 'html')
        """
        if format_type == "console":
            output = DebateFormatter.format_for_console(conversation_state)
            print(output)
        elif format_type == "html":
            output = DebateFormatter.format_for_html(conversation_state)
            # Save HTML output
            html_path = self.persistence.storage_dir / "latest_debate.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.info(f"HTML output saved to: {html_path}")
        else:
            logger.warning(f"Unknown format type: {format_type}")


import os

os.environ["GROQ_API_KEY"] = ""

def main():
    """
    Main entry point demonstrating the infinite debate system.
    
    This example runs a debate on the topic:
    "Would you go back in time and kill baby Hitler?"
    """
    # Get API key from environment
    api_key = get_environment_variable(
        "GROQ_API_KEY",
        required=False
    )
    
    if not api_key:
        logger.error(
            "GROQ_API_KEY environment variable not set. "
            "Please set it before running the system."
        )
        return
    
    # Initialize system
    system = InfiniteDebateSystem(
        api_key=api_key,
        storage_dir="./conversations",
        log_level="INFO",
        log_file="./logs/debate_system.log"
    )
    
    # Define debate topic
    topic = "Would you go back in time and kill baby Hitler?"
    
    # Define agent stances
    agent_1_stance = (
        "I argue that one SHOULD go back in time and kill baby Hitler if given "
        "the opportunity. The prevention of World War II and the Holocaust "
        "justifies this action from a utilitarian perspective."
    )
    
    agent_2_stance = (
        "I argue that one SHOULD NOT go back in time and kill baby Hitler. "
        "This position is morally indefensible, creates unpredictable timeline "
        "consequences, and violates fundamental ethical principles about "
        "punishing individuals for crimes they haven't yet committed."
    )
    
    logger.info("=" * 80)
    logger.info("STARTING INFINITE DEBATE DEMONSTRATION")
    logger.info("=" * 80)
    logger.info(f"Topic: {topic}")
    logger.info(f"Agent 1 Stance: {agent_1_stance[:100]}...")
    logger.info(f"Agent 2 Stance: {agent_2_stance[:100]}...")
    logger.info("=" * 80)
    
    # Run debate using Protocol approach with TTS and mem0 memory
    logger.info("\n### Running debate with A2A Protocol + TTS + mem0 Memory ###\n")
    
    # Configure TTS with different voices for each agent
    tts_config = TTSConfig(
        agent_1_voice="af_bella",  # Female voice for agent 1
        agent_2_voice="am_adam",    # Male voice for agent 2
        sample_rate=24000,
        enable_realtime=True
    )
    
    final_state_protocol = system.run_debate_with_protocol(
        topic=topic,
        max_turns=6,  # Limit turns for demonstration
        agent_1_stance=agent_1_stance,
        agent_2_stance=agent_2_stance,
        temperature=0.9,
        save_results=True,
        enable_tts=True,
        tts_config=tts_config,
        # enable_memory=True,  # ✅ Enable mem0 memory
        memory_user_id_prefix="hitler_debate"
    )
    
    # Display results
    print("\n\n")
    system.display_conversation(final_state_protocol, format_type="console")
    
    # Compute and display metrics
    metrics = final_state_protocol.compute_metrics()
    print("\n" + "=" * 80)
    print("DEBATE METRICS")
    print("=" * 80)
    print(f"Total Turns: {metrics.total_turns}")
    print(f"Agent 1 Turns: {metrics.agent_1_turns}")
    print(f"Agent 2 Turns: {metrics.agent_2_turns}")
    print(f"Total Tokens (estimated): {metrics.total_tokens_estimate}")
    print(f"Duration: {metrics.conversation_duration_seconds:.2f} seconds")
    print(f"Average Utterance Length: {metrics.average_utterance_length:.1f} chars")
    print("=" * 80)
    
    print("\n✨ Features Demonstrated:")
    print("  ✓ Multi-agent debate with A2A protocol")
    print("  ✓ Real-time TTS audio generation (Kokoro)")
    print("  ✓ Semantic memory with mem0 (agents remember context)")
    print("  ✓ Immutable conversation state")
    print("  ✓ Event-driven monitoring")
    print("  ✓ Auto-save to ./audio_outputs/ and ./conversations/")
    
    # Optional: Run with LangGraph workflow
    logger.info("\n### Alternative: LangGraph Workflow approach available ###")
    logger.info("See examples.py for more usage patterns\n")
    
    logger.info("Debate demonstration completed successfully")


if __name__ == "__main__":
    main()
