import logging
import os
from typing import Optional

from agents import AgentFactory, DebateAgent
from conversation_state import ConversationState, ProtocolParameters
from protocol import A2AProtocol, ProtocolBuilder, ProtocolEvent
from workflow import DebateWorkflow, WorkflowBuilder
from tts import DebateTTSGenerator, TTSConfig
from utils import (
    setup_logging,
    ConversationPersistence,
    MetricsTracker,
    validate_api_key,
    get_environment_variable,
    DebateFormatter,
)

from memory import (
    DebateMemoryManager,
    MemoryLayer,
    EpisodeEventType,
    ClaimRelation,
    DEFAULT_TOKEN_BUDGET,
)

logger = logging.getLogger(__name__)


class MemoryAwareEventHandler:
    def __init__(
        self,
        memory_manager: DebateMemoryManager,
        metrics_tracker: MetricsTracker,
    ):
        self._mem     = memory_manager
        self._metrics = metrics_tracker
        # Track last text per agent so we can process the turn after it completes
        self._pending: dict = {}

    def handle(self, event: ProtocolEvent, data: dict) -> None:
        """Dispatch on event type and route to the appropriate memory layer."""
        self._metrics.log_event(event.value, data)

        if event == ProtocolEvent.TURN_COMPLETED:
            self._on_turn_completed(data)

        elif event == ProtocolEvent.CONVERSATION_TERMINATED:
            self._on_conversation_terminated(data)

        if event == ProtocolEvent.TURN_COMPLETED:
            logger.info(
                f"Turn {data.get('turn_number')} completed "
                f"by {data.get('agent_id')}"
            )
        elif event == ProtocolEvent.CONVERSATION_TERMINATED:
            logger.info(f"Conversation terminated: {data.get('reason')}")

    def _on_turn_completed(self, data: dict):
        conv_id    = data.get("conversation_id", "unknown")
        agent_id   = data.get("agent_id", "unknown")
        text       = data.get("text", "")
        turn_index = data.get("turn_number", 0)

        # L1 — push verbatim turn
        self._mem.push_turn(conv_id, agent_id, text, turn_index)

        # L2 — extract claims + detect boundaries + update claim graph
        self._mem.process_turn(conv_id, agent_id, text, turn_index)

        # L4 — heuristically infer and persist a belief from each turn
        if len(text) > 60:
            # Use the first meaningful sentence as a potential belief update
            first_sentence = text.split(".")[0].strip()
            if len(first_sentence) > 30:
                self._mem.update_belief(
                    agent_id    = agent_id,
                    content     = first_sentence,
                    belief_type = "ideological",
                    confidence  = 0.75,
                    conv_id     = conv_id,
                )

    def _on_conversation_terminated(self, data: dict):
        conv_id = data.get("conversation_id", "unknown")
        reason  = data.get("reason", "")

        # Add a global shared fact about the debate outcome
        outcome_note = (
            f"Debate '{data.get('topic', '')}' terminated after "
            f"{data.get('total_turns', '?')} turns. Reason: {reason}."
        )
        self._mem.add_global_fact(
            content    = outcome_note,
            speaker_id = "system",
            conv_id    = conv_id,
            confidence = 1.0,
        )


class MemoryAugmentedDebateAgent:
    def __init__(
        self,
        base_agent:     DebateAgent,
        memory_manager: DebateMemoryManager,
        conv_id:        str,
        token_budget:   int = DEFAULT_TOKEN_BUDGET,
    ):
        self._agent   = base_agent
        self._mem     = memory_manager
        self._conv_id = conv_id
        self._budget  = token_budget

    # Delegate attribute access to base agent for protocol compatibility
    def __getattr__(self, name):
        return getattr(self._agent, name)

    def generate_response(
        self, conversation_history: list, current_topic: str, **kwargs
    ) -> str:
        """
        1. Retrieve hierarchical memory context.
        2. Prepend context block to the system / user prompt.
        3. Delegate to base agent for actual LLM call.
        """
        # Construct query from latest history entry
        query = current_topic
        if conversation_history:
            last = conversation_history[-1]
            query = last.get("content", current_topic)

        # Retrieve context under token budget
        ctx = self._mem.retrieve_context(
            conv_id      = self._conv_id,
            agent_id     = self._agent.agent_id,
            current_text = query,
            token_budget = self._budget,
        )

        memory_block = ctx.to_prompt_block()

        # Inject into the system message (first entry) or prepend a new one
        augmented_history = list(conversation_history)
        if memory_block.strip():
            memory_msg = {
                "role": "system",
                "content": (
                    "=== MEMORY CONTEXT (use this to maintain coherence) ===\n"
                    + memory_block
                    + "\n=== END MEMORY CONTEXT ==="
                ),
            }
            # Insert after initial system prompt if present, else prepend
            if augmented_history and augmented_history[0]["role"] == "system":
                augmented_history.insert(1, memory_msg)
            else:
                augmented_history.insert(0, memory_msg)

        logger.debug(
            f"[{self._agent.agent_id}] memory context injected: "
            f"~{ctx.token_estimate} tokens"
        )

        return self._agent.generate_response(
            augmented_history, current_topic, **kwargs
        )


class InfiniteDebateSystem:
    def __init__(
        self,
        api_key:     str,
        storage_dir: str           = "./conversations",
        memory_db:   str           = "./memory_store/debate_memory.db",
        log_level:   str           = "INFO",
        log_file:    Optional[str] = None,
    ):
        setup_logging(log_level=log_level, log_file=log_file)

        if not validate_api_key(api_key):
            raise ValueError("Invalid API key format")

        self.api_key         = api_key
        self.persistence     = ConversationPersistence(storage_dir)
        self.metrics_tracker = MetricsTracker()

        self.memory_manager = DebateMemoryManager(
            db_path         = memory_db,
            user_id_prefix  = "debate",
            compression_every = 6,
        )
        logger.info("DebateMemoryManager (5-layer) initialised")

    def run_debate_with_protocol(
        self,
        topic:              str,
        max_turns:          Optional[int] = 10,
        agent_1_stance:     Optional[str] = None,
        agent_2_stance:     Optional[str] = None,
        temperature:        float          = 1.0,
        save_results:       bool           = True,
        enable_tts:         bool           = False,
        tts_config:         Optional[TTSConfig] = None,
        token_budget:       int            = DEFAULT_TOKEN_BUDGET,
        seed_global_facts:  Optional[list] = None,
    ) -> ConversationState:
        import uuid as _uuid
        conv_id = f"conv_{_uuid.uuid4().hex[:8]}"

        logger.info(
            f"Starting debate (Protocol + Memory) | "
            f"topic='{topic}' | conv_id={conv_id}"
        )

        # Seed global shared memory (L5) 
        if seed_global_facts:
            for fact in seed_global_facts:
                self.memory_manager.add_global_fact(
                    content=fact, speaker_id="system", conv_id=conv_id
                )
            logger.info(f"Seeded {len(seed_global_facts)} global facts into L5")

        # Seed debate topic as a global fact 
        self.memory_manager.add_global_fact(
            content    = f"Current debate topic: {topic}",
            speaker_id = "system",
            conv_id    = conv_id,
            confidence = 1.0,
        )

        # Build memory-aware event handler
        event_handler = MemoryAwareEventHandler(
            memory_manager  = self.memory_manager,
            metrics_tracker = self.metrics_tracker,
        )

        # Create base debate agents
        base_agent_1 = DebateAgent(
            agent_id    = "agent_1",
            api_key     = self.api_key,
            stance      = agent_1_stance or "Support the affirmative position",
            personality = (
                "Analytical and evidence-focused, "
                "favouring logical arguments and empirical data"
            ),
        )
        base_agent_2 = DebateAgent(
            agent_id    = "agent_2",
            api_key     = self.api_key,
            stance      = agent_2_stance or "Support the negative position",
            personality = (
                "Philosophical and principle-based, "
                "emphasising ethical considerations and thought experiments"
            ),
        )

        # Wrap with memory augmentation
        agent_1 = MemoryAugmentedDebateAgent(
            base_agent     = base_agent_1,
            memory_manager = self.memory_manager,
            conv_id        = conv_id,
            token_budget   = token_budget,
        )
        agent_2 = MemoryAugmentedDebateAgent(
            base_agent     = base_agent_2,
            memory_manager = self.memory_manager,
            conv_id        = conv_id,
            token_budget   = token_budget,
        )

        logger.info("Memory-augmented agents created")

        # Seed initial agent beliefs (L4)
        if agent_1_stance:
            self.memory_manager.update_belief(
                agent_id    = "agent_1",
                content     = agent_1_stance,
                belief_type = "ideological",
                confidence  = 1.0,
                conv_id     = conv_id,
            )
        if agent_2_stance:
            self.memory_manager.update_belief(
                agent_id    = "agent_2",
                content     = agent_2_stance,
                belief_type = "ideological",
                confidence  = 1.0,
                conv_id     = conv_id,
            )
        logger.info("Initial agent beliefs persisted in L4")

        protocol = ProtocolBuilder.create_standard_protocol(
            agent_1        = agent_1,
            agent_2        = agent_2,
            topic          = topic,
            max_turns      = max_turns,
            temperature    = temperature,
            event_callback = event_handler.handle,
            enable_tts     = enable_tts,
            tts_config     = tts_config,
        )

        final_state = protocol.run_conversation()

        analytics = self.memory_manager.get_debate_analytics(conv_id)
        logger.info(
            f"Memory analytics: {analytics['total_claims']} claims | "
            f"{analytics['total_episodes']} episodes | "
            f"{analytics['unresolved_count']} unresolved"
        )

        narrative = self.memory_manager.reconstruct_episode_narrative(conv_id)
        if narrative:
            logger.info(f"Episode narrative:\n{narrative}")

        if enable_tts and protocol.tts_generator:
            try:
                audio_path = protocol.tts_generator.save_debate_audio(topic=topic)
                logger.info(f"Saved debate audio: {audio_path}")
                stats = protocol.tts_generator.get_statistics()
                logger.info(
                    f"TTS Stats: {stats['total_duration']:.2f}s | "
                    f"{stats['agent_1_turns']} + {stats['agent_2_turns']} turns"
                )
            except Exception as exc:
                logger.error(f"Failed to save TTS audio: {exc}", exc_info=True)

        if save_results:
            self.persistence.save_conversation(final_state)
            self.persistence.save_transcript(final_state)

        logger.info(
            f"Debate completed — {final_state.current_turn_number} turns"
        )
        return final_state

    def run_debate_with_workflow(
        self,
        topic:          str,
        max_turns:      Optional[int] = 10,
        agent_1_stance: Optional[str] = None,
        agent_2_stance: Optional[str] = None,
        temperature:    float          = 1.0,
        save_results:   bool           = True,
    ) -> ConversationState:
        logger.info(f"Starting debate (Workflow + Memory) | topic='{topic}'")

        agent_1, agent_2 = AgentFactory.create_debate_pair(
            api_key        = self.api_key,
            topic          = topic,
            agent_1_stance = agent_1_stance,
            agent_2_stance = agent_2_stance,
        )

        protocol_params    = ProtocolParameters(max_turns=max_turns, temperature=temperature)
        conversation_state = ConversationState(topic=topic, protocol_params=protocol_params)

        workflow = WorkflowBuilder.create_standard_workflow(
            agent_1            = agent_1,
            agent_2            = agent_2,
            conversation_state = conversation_state,
        )

        final_state = workflow.run(max_turns=max_turns)

        if save_results:
            self.persistence.save_conversation(final_state)
            self.persistence.save_transcript(final_state)

        logger.info(
            f"Debate completed — {final_state.current_turn_number} turns"
        )
        return final_state

    def display_conversation(
        self,
        conversation_state: ConversationState,
        format_type:        str = "console",
    ) -> None:
        if format_type == "console":
            print(DebateFormatter.format_for_console(conversation_state))
        elif format_type == "html":
            output   = DebateFormatter.format_for_html(conversation_state)
            html_path = self.persistence.storage_dir / "latest_debate.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(output)
            logger.info(f"HTML output saved to: {html_path}")
        else:
            logger.warning(f"Unknown format type: {format_type}")

    def display_memory_report(self, conv_id: str) -> None:
        """Print a structured report of the debate's memory state."""
        analytics = self.memory_manager.get_debate_analytics(conv_id)
        narrative = self.memory_manager.reconstruct_episode_narrative(conv_id)

        print("\n" + "=" * 80)
        print("HIERARCHICAL MEMORY REPORT")
        print("=" * 80)
        print(f"Conversation ID   : {analytics['conversation_id']}")
        print(f"Total Claims (L2) : {analytics['total_claims']}")
        print(f"Total Episodes    : {analytics['total_episodes']}")
        print(f"Unresolved Issues : {analytics['unresolved_count']}")
        print(f"Semantic Facts L3 : {analytics['semantic_memory_count']}")
        print(f"Global Facts   L5 : {analytics['global_memory_count']}")
        print()
        print("Speaker Distribution:")
        for speaker, count in analytics["speaker_distribution"].items():
            print(f"  {speaker}: {count} claims")
        print()
        print("Episode Type Breakdown:")
        for etype, count in analytics["event_type_distribution"].items():
            print(f"  {etype}: {count}")
        print()
        print("Episode Narrative:")
        print(narrative or "  (none)")
        print("=" * 80)



# os.environ["GROQ_API_KEY"] = ""   # set your key here or via env


def main():
    api_key = get_environment_variable("GROQ_API_KEY", required=False)

    if not api_key:
        logger.error(
            "GROQ_API_KEY not set. "
            "Export it or assign it in os.environ above."
        )
        return

    system = InfiniteDebateSystem(
        api_key     = api_key,
        storage_dir = "./conversations",
        memory_db   = "./memory_store/debate_memory.db",
        log_level   = "INFO",
        log_file    = "./logs/debate_system.log",
    )

    topic = "Would you go back in time and kill baby Hitler?"

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

    # Background world-knowledge seeded into L5 (Global Shared Memory)
    seed_facts = [
        "World War II (1939–1945) resulted in an estimated 70–85 million deaths.",
        "The Holocaust caused the systematic murder of approximately 6 million Jews.",
        "The trolley problem is a classic ethical thought experiment in moral philosophy.",
        (
            "The grandfather paradox questions whether killing an ancestor creates "
            "a causal contradiction in time-travel scenarios."
        ),
        (
            "Utilitarian ethics evaluates actions based on aggregate outcomes; "
            "deontological ethics evaluates actions based on rules and duties."
        ),
    ]

    logger.info(f"Topic : {topic}")
    logger.info(f"Layer : L1=STM | L2=Episodic | L3=Semantic | L4=Beliefs | L5=Global")

    tts_config = TTSConfig(
        agent_1_voice  = "af_bella",
        agent_2_voice  = "am_adam",
        sample_rate    = 24000,
        enable_realtime = True,
    )

    final_state = system.run_debate_with_protocol(
        topic             = topic,
        max_turns         = 2,          # increase for longer debates
        agent_1_stance    = agent_1_stance,
        agent_2_stance    = agent_2_stance,
        temperature       = 0.9,
        save_results      = True,
        enable_tts        = True,
        tts_config        = tts_config,
        token_budget      = DEFAULT_TOKEN_BUDGET,
        seed_global_facts = seed_facts,
    )

    print("\n\n")
    system.display_conversation(final_state, format_type="console")

    metrics = final_state.compute_metrics()
    print("\n" + "=" * 80)
    print("DEBATE METRICS")
    print("=" * 80)
    print(f"Total Turns               : {metrics.total_turns}")
    print(f"Agent 1 Turns             : {metrics.agent_1_turns}")
    print(f"Agent 2 Turns             : {metrics.agent_2_turns}")
    print(f"Total Tokens (estimated)  : {metrics.total_tokens_estimate}")
    print(f"Duration                  : {metrics.conversation_duration_seconds:.2f}s")
    print(f"Avg Utterance Length      : {metrics.average_utterance_length:.1f} chars")
    print("=" * 80)

    # Retrieve the conv_id from state if available, otherwise use placeholder
    conv_id = getattr(final_state, "conversation_id", "unknown")
    system.display_memory_report(conv_id)


if __name__ == "__main__":
    main()
