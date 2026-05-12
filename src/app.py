import gradio as gr
import logging
import os
import uuid
from typing import Optional, Generator, Tuple

from agents import DebateAgent
from conversation_state import ConversationState, ProtocolParameters
from protocol import ProtocolBuilder, ProtocolEvent, A2AProtocol
from tts import DebateTTSGenerator, TTSConfig
from utils import setup_logging, ConversationPersistence, MetricsTracker, validate_api_key, DebateFormatter
from memory import DebateMemoryManager, DEFAULT_TOKEN_BUDGET

os.makedirs("./logs", exist_ok=True)
os.makedirs("./conversations", exist_ok=True)
os.makedirs("./memory_store", exist_ok=True)
os.makedirs("./audio_outputs", exist_ok=True)

setup_logging(log_level="INFO", log_file="./logs/gradio_app.log")
logger = logging.getLogger(__name__)


class MemoryAwareEventHandler:
    def __init__(self, memory_manager: DebateMemoryManager, metrics_tracker: MetricsTracker):
        self._mem     = memory_manager
        self._metrics = metrics_tracker
        self._pending: dict = {}

    def handle(self, event: ProtocolEvent, data: dict) -> None:
        self._metrics.log_event(event.value, data)
        if event == ProtocolEvent.TURN_COMPLETED:
            self._on_turn_completed(data)
            logger.info(f"Turn {data.get('turn_number')} completed by {data.get('agent_id')}")
        elif event == ProtocolEvent.CONVERSATION_TERMINATED:
            self._on_conversation_terminated(data)
            logger.info(f"Conversation terminated: {data.get('reason')}")

    def _on_turn_completed(self, data: dict):
        conv_id    = data.get("conversation_id", "unknown")
        agent_id   = data.get("agent_id", "unknown")
        text       = data.get("text", "")
        turn_index = data.get("turn_number", 0)

        self._mem.push_turn(conv_id, agent_id, text, turn_index)
        self._mem.process_turn(conv_id, agent_id, text, turn_index)

        if len(text) > 60:
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
        conv_id      = data.get("conversation_id", "unknown")
        reason       = data.get("reason", "")
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
    def __init__(self, base_agent: DebateAgent, memory_manager: DebateMemoryManager,
                 conv_id: str, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self._agent   = base_agent
        self._mem     = memory_manager
        self._conv_id = conv_id
        self._budget  = token_budget

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def generate_response(self, conversation_history: list, current_topic: str, **kwargs) -> str:
        query = current_topic
        if conversation_history:
            last  = conversation_history[-1]
            query = last.get("content", current_topic)

        ctx          = self._mem.retrieve_context(
            conv_id      = self._conv_id,
            agent_id     = self._agent.agent_id,
            current_text = query,
            token_budget = self._budget,
        )
        memory_block = ctx.to_prompt_block()

        augmented_history = list(conversation_history)
        if memory_block.strip():
            memory_msg = {
                "role":    "system",
                "content": (
                    "=== MEMORY CONTEXT (use this to maintain coherence) ===\n"
                    + memory_block
                    + "\n=== END MEMORY CONTEXT ==="
                ),
            }
            if augmented_history and augmented_history[0]["role"] == "system":
                augmented_history.insert(1, memory_msg)
            else:
                augmented_history.insert(0, memory_msg)

        logger.debug(f"[{self._agent.agent_id}] memory context ~{ctx.token_estimate} tokens")
        return self._agent.generate_response(augmented_history, current_topic, **kwargs)


GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.3-70b-versatile",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

FEMALE_VOICES = ["af_bella", "af_sarah", "af_nicole", "af_sky", "af_heart"]
MALE_VOICES   = ["am_adam", "am_michael", "am_eric"]

DEFAULT_SEED_FACTS = [
    "World War II (1939–1945) resulted in an estimated 70–85 million deaths.",
    "The Holocaust caused the systematic murder of approximately 6 million Jews.",
    "The trolley problem is a classic ethical thought experiment in moral philosophy.",
    "The grandfather paradox questions whether killing an ancestor creates a causal contradiction in time-travel scenarios.",
    "Utilitarian ethics evaluates actions based on aggregate outcomes; deontological ethics evaluates actions based on rules and duties.",
]


def _format_chat(history: list, topic: str = None) -> str:
    """Render conversation history as styled HTML."""
    if not history:
        placeholder = (
            f"<div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);"
            f"padding:20px;border-radius:12px;margin-bottom:20px;'>"
            f"<h2 style='margin:0;color:#e2e8f0;font-size:1.2em;'>🎭 Debate Topic</h2>"
            f"<p style='margin:10px 0 0;color:#94a3b8;font-size:1.05em;line-height:1.6;'>{topic}</p></div>"
            if topic else ""
        )
        return (
            f"<div style='font-family:\"IBM Plex Mono\",monospace;'>{placeholder}"
            "<div style='text-align:center;padding:40px;color:#64748b;'>"
            "<p style='font-size:1.1em;'>⏳ Waiting for agents to begin…</p></div></div>"
        )

    parts = []

    if topic:
        parts.append(
            f"<div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);"
            f"padding:20px;border-radius:12px;margin-bottom:20px;border:1px solid #334155;'>"
            f"<h2 style='margin:0;color:#e2e8f0;font-size:1.2em;font-family:\"IBM Plex Mono\",monospace;'>🎭 Debate Topic</h2>"
            f"<p style='margin:10px 0 0;color:#94a3b8;font-size:1.05em;line-height:1.6;"
            f"font-family:\"IBM Plex Mono\",monospace;'>{topic}</p></div>"
        )

    parts.append(
        "<div id='debate-container' style='font-family:\"IBM Plex Mono\",monospace;"
        "max-height:580px;overflow-y:auto;padding:8px;'>"
    )

    for msg in history:
        is_agent1  = msg["agent_id"] == "agent_1"
        num        = "1" if is_agent1 else "2"
        color      = "#38bdf8" if is_agent1 else "#4ade80"
        bg         = "#0f172a" if is_agent1 else "#052e16"
        border     = "#0ea5e9" if is_agent1 else "#16a34a"
        emoji      = "🔵" if is_agent1 else "🟢"
        align      = "flex-start" if is_agent1 else "flex-end"

        parts.append(
            f"<div style='margin:12px 0;padding:14px 16px;background:{bg};"
            f"border:1px solid {border};border-radius:10px;"
            f"display:flex;flex-direction:column;align-items:{align};'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;width:100%;justify-content:{align};'>"
            f"<span style='font-size:1em;'>{emoji}</span>"
            f"<span style='font-weight:700;color:{color};font-size:0.88em;letter-spacing:.05em;'>AGENT {num}</span>"
            f"<span style='margin-left:auto;color:#475569;font-size:0.78em;'>Turn {msg['turn_number']}</span>"
            f"</div>"
            f"<div style='color:#cbd5e1;line-height:1.75;font-size:0.87em;width:100%;'>"
            f"{msg['utterance']}</div></div>"
        )

    parts.append(
        "</div>"
        "<style>"
        "#debate-container::-webkit-scrollbar{width:6px}"
        "#debate-container::-webkit-scrollbar-track{background:#0f172a;border-radius:6px}"
        "#debate-container::-webkit-scrollbar-thumb{background:#334155;border-radius:6px}"
        "</style>"
        "<script>"
        "const c=document.getElementById('debate-container');"
        "if(c)c.scrollTop=c.scrollHeight;"
        "</script>"
    )
    return "".join(parts)


def build_memory_report(memory_manager: DebateMemoryManager, conv_id: str) -> str:
    analytics = memory_manager.get_debate_analytics(conv_id)
    narrative = memory_manager.reconstruct_episode_narrative(conv_id)

    lines = [
        "=" * 60,
        "  HIERARCHICAL MEMORY REPORT",
        "=" * 60,
        f"  Conversation ID   : {analytics['conversation_id']}",
        f"  Total Claims (L2) : {analytics['total_claims']}",
        f"  Total Episodes    : {analytics['total_episodes']}",
        f"  Unresolved Issues : {analytics['unresolved_count']}",
        f"  Semantic Facts L3 : {analytics['semantic_memory_count']}",
        f"  Global Facts   L5 : {analytics['global_memory_count']}",
        "",
        "  Speaker Distribution:",
    ]
    for speaker, count in analytics["speaker_distribution"].items():
        lines.append(f"    {speaker}: {count} claims")
    lines.append("")
    lines.append("  Episode Type Breakdown:")
    for etype, count in analytics["event_type_distribution"].items():
        lines.append(f"    {etype}: {count}")
    lines.append("")
    lines.append("  Episode Narrative:")
    lines.append(f"  {narrative or '(none)'}")
    lines.append("=" * 60)
    return "\n".join(lines)


def run_debate_streaming(
    api_key: str,
    topic: str,
    max_turns: int,
    # Agent 1
    agent1_stance: str,
    agent1_personality: str,
    agent1_temperature: float,
    # Agent 2
    agent2_stance: str,
    agent2_personality: str,
    agent2_temperature: float,
    # Model
    model_name: str,
    max_tokens: int,
    # Memory
    token_budget: int,
    seed_facts_text: str,
    # TTS
    enable_tts: bool,
    agent1_voice: str,
    agent2_voice: str,
    # Misc
    save_results: bool,
    temperature: float,
) -> Generator[Tuple[str, str, str, Optional[str]], None, None]:
    """
    Generator that yields (chat_html, status_text, memory_report, audio_path)
    after every turn.
    """
    # ── Validation ──────────────────────────────────────────────────────────
    if not topic.strip():
        yield _format_chat([]), "❌ Topic cannot be empty.", "", None
        return
    if not api_key or not api_key.strip():
        yield _format_chat([]), "❌ GROQ_API_KEY not set. Add it in the API Key field.", "", None
        return

    conv_id = f"conv_{uuid.uuid4().hex[:8]}"
    conversation_history: list = []

    yield _format_chat([], topic=topic), "🔄 Initialising memory system…", "", None

    # ── Memory manager ───────────────────────────────────────────────────────
    memory_manager = DebateMemoryManager(
        db_path           = "./memory_store/debate_memory.db",
        user_id_prefix    = "debate",
        compression_every = 6,
    )

    # Seed global facts (L5)
    seed_facts = [line.strip() for line in seed_facts_text.strip().splitlines() if line.strip()]
    for fact in seed_facts:
        memory_manager.add_global_fact(content=fact, speaker_id="system", conv_id=conv_id)
    memory_manager.add_global_fact(
        content    = f"Current debate topic: {topic}",
        speaker_id = "system",
        conv_id    = conv_id,
        confidence = 1.0,
    )
    logger.info(f"Seeded {len(seed_facts)} global facts into L5 | conv_id={conv_id}")

    metrics_tracker = MetricsTracker()
    event_handler   = MemoryAwareEventHandler(memory_manager, metrics_tracker)

    # ── Base agents ──────────────────────────────────────────────────────────
    base_agent_1 = DebateAgent(
        agent_id    = "agent_1",
        api_key     = api_key.strip(),
        stance      = agent1_stance or "Support the affirmative position",
        personality = agent1_personality or "Analytical and evidence-focused",
    )
    base_agent_2 = DebateAgent(
        agent_id    = "agent_2",
        api_key     = api_key.strip(),
        stance      = agent2_stance or "Support the negative position",
        personality = agent2_personality or "Philosophical and principle-based",
    )

    # Seed initial beliefs (L4)
    if agent1_stance:
        memory_manager.update_belief(
            agent_id="agent_1", content=agent1_stance,
            belief_type="ideological", confidence=1.0, conv_id=conv_id,
        )
    if agent2_stance:
        memory_manager.update_belief(
            agent_id="agent_2", content=agent2_stance,
            belief_type="ideological", confidence=1.0, conv_id=conv_id,
        )

    # ── Memory-augmented wrappers ────────────────────────────────────────────
    agent_1 = MemoryAugmentedDebateAgent(base_agent_1, memory_manager, conv_id, token_budget)
    agent_2 = MemoryAugmentedDebateAgent(base_agent_2, memory_manager, conv_id, token_budget)

    # ── Conversation state ───────────────────────────────────────────────────
    protocol_params    = ProtocolParameters(
        max_turns             = max_turns,
        temperature           = temperature,
        max_utterance_length  = max_tokens,
        model                 = model_name,
    )
    conversation_state = ConversationState(topic=topic, protocol_params=protocol_params)

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_generator = None
    if enable_tts:
        tts_config    = TTSConfig(
            agent_1_voice   = agent1_voice,
            agent_2_voice   = agent2_voice,
            sample_rate     = 24000,
            enable_realtime = True,
        )
        tts_generator = DebateTTSGenerator(config=tts_config, output_dir="./audio_outputs")

    # ── Protocol ─────────────────────────────────────────────────────────────
    protocol = ProtocolBuilder.create_standard_protocol(
        agent_1        = agent_1,
        agent_2        = agent_2,
        topic          = topic,
        max_turns      = max_turns,
        temperature    = temperature,
        event_callback = event_handler.handle,
        enable_tts     = enable_tts,
        tts_config     = tts_config if enable_tts else None,
    )

    yield _format_chat([], topic=topic), f"🎭 Debate started: {topic}", "", None

    # ── Turn loop ─────────────────────────────────────────────────────────────
    import time

    turn = 0
    while not conversation_state.is_terminated() and turn < max_turns:
        current_agent = protocol.agents_map[conversation_state.next_speaker]
        agent_label   = "Agent 1" if current_agent.agent_id == "agent_1" else "Agent 2"
        agent_temp    = agent1_temperature if current_agent.agent_id == "agent_1" else agent2_temperature

        yield (
            _format_chat(conversation_history, topic=topic),
            f"🎤 {agent_label} is thinking… (Turn {turn + 1}/{max_turns})",
            "",
            None,
        )

        try:
            context    = conversation_state.get_context_for_agent(conversation_state.next_speaker)
            start_time = time.time()

            utterance = current_agent.generate_utterance(
                context     = context,
                temperature = agent_temp,
                model       = model_name,
                max_tokens  = max_tokens,
            )
            gen_time = time.time() - start_time

            conversation_state.append_utterance(current_agent.agent_id, utterance)
            conversation_history.append({
                "agent_id":    current_agent.agent_id,
                "utterance":   utterance,
                "turn_number": conversation_state.current_turn_number,
            })

            status = f"✓ {agent_label} responded — Turn {turn + 1}/{max_turns} ({gen_time:.1f}s)"

            # TTS
            audio_path = None
            if enable_tts and tts_generator:
                tts_start = time.time()
                tts_generator.add_utterance(
                    text        = utterance,
                    agent_id    = current_agent.agent_id,
                    turn_number = conversation_state.current_turn_number,
                    add_pause   = True,
                )
                status += f" | TTS {time.time()-tts_start:.1f}s"

            turn += 1
            yield _format_chat(conversation_history, topic=topic), status, "", audio_path

        except Exception as exc:
            logger.error(f"Error in turn {turn}: {exc}", exc_info=True)
            yield (
                _format_chat(conversation_history, topic=topic),
                f"❌ Error in turn {turn + 1}: {exc}",
                "",
                None,
            )
            break

    # ── Finalise ──────────────────────────────────────────────────────────────
    metrics = conversation_state.compute_metrics()
    final_status = (
        f"✅ Debate complete!\n\n"
        f"📊 Turns: {turn}  |  Duration: {metrics.conversation_duration_seconds:.1f}s  |  "
        f"Agent 1: {metrics.agent_1_turns} turns  |  Agent 2: {metrics.agent_2_turns} turns"
    )

    memory_report = build_memory_report(memory_manager, conv_id)

    # Analytics log
    analytics = memory_manager.get_debate_analytics(conv_id)
    logger.info(
        f"Memory analytics: {analytics['total_claims']} claims | "
        f"{analytics['total_episodes']} episodes | "
        f"{analytics['unresolved_count']} unresolved"
    )

    # Persistence
    if save_results:
        persistence = ConversationPersistence("./conversations")
        persistence.save_conversation(conversation_state)
        persistence.save_transcript(conversation_state)

    # Final audio
    audio_path = None
    if enable_tts and tts_generator:
        try:
            audio_path   = tts_generator.save_debate_audio(topic=topic)
            stats        = tts_generator.get_statistics()
            final_status += f"\n🎵 Audio: {stats['total_duration']:.1f}s saved to {audio_path}"
        except Exception as exc:
            logger.error(f"Failed to save TTS audio: {exc}", exc_info=True)
            final_status += f"\n⚠️ Audio save failed: {exc}"

    yield _format_chat(conversation_history, topic=topic), final_status, memory_report, audio_path


# ── Interface ─────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    api_key_env = os.environ.get("GROQ_API_KEY", "")

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    body, .gradio-container {
        background: #020817 !important;
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .gradio-container { max-width: 1400px !important; }

    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }

    .panel-box {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    label { color: #94a3b8 !important; font-size: 0.82em !important; letter-spacing: .06em !important; text-transform: uppercase !important; }

    input, textarea, select {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    .gr-button-primary {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        border: none !important;
        color: white !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: .08em !important;
    }
    .gr-button-stop {
        background: #7f1d1d !important;
        border: 1px solid #991b1b !important;
        color: #fca5a5 !important;
    }
    """

    default_seed = "\n".join(DEFAULT_SEED_FACTS)

    with gr.Blocks(css=custom_css, title="Infinite Debate System") as app:

        gr.HTML("""
        <div style='text-align:center;padding:32px 0 16px;font-family:"IBM Plex Mono",monospace;'>
            <div style='font-size:2.4em;font-weight:700;background:linear-gradient(135deg,#38bdf8,#818cf8);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:.04em;'>
                Logorrhea: Multi Agent AI Debate System
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            with gr.Column(scale=3):
                chat_display = gr.HTML(
                    value=_format_chat([]),
                    label="Live Debate",
                )

                status_display = gr.Textbox(
                    label="Status",
                    value="Configure settings →  then click START DEBATE",
                    interactive=False,
                    lines=3,
                )

                memory_display = gr.Textbox(
                    label="Memory Report (post-debate)",
                    interactive=False,
                    lines=14,
                    placeholder="Memory analytics will appear here after the debate completes.",
                )

                audio_output = gr.Audio(
                    label="Debate Audio",
                    visible=True,
                    type="filepath",
                )

            # ── Right column: config panel ─────────────────────────────────────
            with gr.Column(scale=2):

                with gr.Accordion("🔑 API Key", open=not bool(api_key_env)):
                    api_key_input = gr.Textbox(
                        label="GROQ API Key",
                        placeholder="gsk_…",
                        value=api_key_env,
                        type="password",
                        lines=1,
                    )

                with gr.Accordion("📝 Debate Setup", open=True):
                    topic_input = gr.Textbox(
                        label="Debate Topic",
                        value="Would you go back in time and kill baby Hitler?",
                        lines=2,
                    )
                    with gr.Row():
                        max_turns_slider = gr.Slider(label="Max Turns", minimum=2, maximum=30, value=6, step=1)
                        temperature      = gr.Slider(label="Global Temperature", minimum=0.1, maximum=2.0, value=0.9, step=0.05)

                    with gr.Row():
                        model_select = gr.Dropdown(
                            label="LLM Model", choices=GROQ_MODELS, value="llama-3.1-8b-instant"
                        )
                        max_tokens = gr.Slider(label="Max Tokens", minimum=128, maximum=4096, value=1024, step=128)

                    save_results = gr.Checkbox(label="Save transcript & conversation JSON", value=True)

                with gr.Accordion("🤖 Agent 1  (Affirmative)", open=False):
                    a1_stance = gr.Textbox(
                        label="Stance",
                        value=(
                            "I argue that one SHOULD go back in time and kill baby Hitler if given "
                            "the opportunity. The prevention of World War II and the Holocaust "
                            "justifies this action from a utilitarian perspective."
                        ),
                        lines=3,
                    )
                    a1_personality = gr.Textbox(
                        label="Personality",
                        value="Analytical and evidence-focused, favouring logical arguments and empirical data",
                        lines=2,
                    )
                    a1_temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.9, step=0.05)

                with gr.Accordion("🤖 Agent 2  (Negative)", open=False):
                    a2_stance = gr.Textbox(
                        label="Stance",
                        value=(
                            "I argue that one SHOULD NOT go back in time and kill baby Hitler. "
                            "This position is morally indefensible, creates unpredictable timeline "
                            "consequences, and violates fundamental ethical principles about "
                            "punishing individuals for crimes they haven't yet committed."
                        ),
                        lines=3,
                    )
                    a2_personality = gr.Textbox(
                        label="Personality",
                        value="Philosophical and principle-based, emphasising ethical considerations and thought experiments",
                        lines=2,
                    )
                    a2_temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.9, step=0.05)

                with gr.Accordion("🧠 Memory Settings", open=False):
                    token_budget = gr.Slider(
                        label="Memory Token Budget",
                        minimum=256,
                        maximum=4096,
                        value=DEFAULT_TOKEN_BUDGET,
                        step=128,
                    )
                    seed_facts_input = gr.TextArea(
                        label="Seed Global Facts (L5) — one per line",
                        value=default_seed,
                        lines=6,
                    )
                    gr.Markdown(
                        "_These facts are seeded into the Global Shared Memory layer "
                        "and are available to both agents throughout the debate._",
                        elem_classes=[]
                    )

                with gr.Accordion("🎤 Text-to-Speech", open=False):
                    enable_tts = gr.Checkbox(label="Enable TTS Audio", value=False)
                    with gr.Row():
                        a1_voice = gr.Dropdown(
                            label="Agent 1 Voice",
                            choices=FEMALE_VOICES + MALE_VOICES,
                            value="af_bella",
                        )
                        a2_voice = gr.Dropdown(
                            label="Agent 2 Voice",
                            choices=FEMALE_VOICES + MALE_VOICES,
                            value="am_adam",
                        )

                with gr.Row():
                    start_btn = gr.Button("▶  START DEBATE", variant="primary", size="lg")
                    stop_btn  = gr.Button("⏹  STOP",        variant="stop",    size="lg")

                with gr.Accordion("💡 Preset Topics", open=False):
                    gr.Markdown("""
| Topic | Agent 1 | Agent 2 |
|---|---|---|
| Is free will an illusion? | Determinist | Compatibilist |
| Should AI be regulated? | Pro-regulation | Pro-innovation |
| Is eating meat ethical? | Vegan ethics | Natural order |
| Will AI replace human jobs? | Technological optimist | Labour protectionist |
| Should we colonise Mars? | Expansionist | Earth-first |
                    """)

        start_btn.click(
            fn=run_debate_streaming,
            inputs=[
                api_key_input,
                topic_input,
                max_turns_slider,
                a1_stance, a1_personality, a1_temperature,
                a2_stance, a2_personality, a2_temperature,
                model_select, max_tokens,
                token_budget, seed_facts_input,
                enable_tts, a1_voice, a2_voice,
                save_results, temperature,
            ],
            outputs=[chat_display, status_display, memory_display, audio_output],
        )

    return app

def main():
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not found in environment — enter it in the UI.")
        print("\n⚠  GROQ_API_KEY not set. You can enter it directly in the app UI.\n")

    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()