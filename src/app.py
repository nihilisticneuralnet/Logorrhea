"""
Gradio Frontend for Infinite Debate System

A minimalist web interface for running and viewing AI debates with
customizable agents, models, and TTS configuration.
"""

import gradio as gr
import logging
import os
from typing import Optional, Tuple, Generator
from datetime import datetime
from pathlib import Path

from agents import AgentFactory, DebateAgent
from conversation_state import ConversationState, ProtocolParameters
from protocol import ProtocolBuilder, ProtocolEvent, A2AProtocol
from tts import DebateTTSGenerator, TTSConfig
from utils import setup_logging, DebateFormatter
from config import ConfigLoader

# Setup logging
setup_logging(log_level="INFO", log_file="./logs/gradio_app.log")
logger = logging.getLogger(__name__)

# Global state for current debate
current_protocol: Optional[A2AProtocol] = None
current_tts_generator: Optional[DebateTTSGenerator] = None


class DebateRunner:
    """Handles debate execution with streaming updates."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.conversation_history = []
    
    def run_debate_streaming(
        self,
        # Basic settings
        topic: str,
        max_turns: int,
        # Agent 1 settings
        agent1_stance: str,
        agent1_personality: str,
        agent1_temperature: float,
        agent1_system_prompt: str,
        # Agent 2 settings
        agent2_stance: str,
        agent2_personality: str,
        agent2_temperature: float,
        agent2_system_prompt: str,
        # Model settings
        model_name: str,
        max_tokens: int,
        # TTS settings
        enable_tts: bool,
        agent1_voice: str,
        agent2_voice: str,
        # Advanced (if implemented)
        turn_timeout: int = 30
    ) -> Generator[Tuple[str, str, Optional[str]], None, None]:
        """
        Run debate with streaming updates.
        
        Yields:
            Tuple of (chat_html, status_text, audio_path)
        """
        global current_protocol, current_tts_generator
        
        try:
            # Validate inputs
            if not topic.strip():
                yield self._format_chat([]), "❌ Error: Topic cannot be empty", None
                return
            
            if not self.api_key:
                yield self._format_chat([]), "❌ Error: GROQ_API_KEY not set", None
                return
            
            # Initialize
            yield self._format_chat([], topic=topic), "🔄 Initializing debate system...", None
            
            # Create agents with custom configurations
            agent_1 = DebateAgent(
                agent_id="agent_1",
                api_key=self.api_key,
                stance=agent1_stance or None,
                personality=agent1_personality or None,
                system_prompt_template=agent1_system_prompt if agent1_system_prompt.strip() else None
            )
            
            agent_2 = DebateAgent(
                agent_id="agent_2",
                api_key=self.api_key,
                stance=agent2_stance or None,
                personality=agent2_personality or None,
                system_prompt_template=agent2_system_prompt if agent2_system_prompt.strip() else None
            )
            
            # Setup protocol parameters
            protocol_params = ProtocolParameters(
                max_turns=max_turns,
                temperature=1.0,  # Will be overridden per agent
                max_utterance_length=max_tokens,
                model=model_name,
                turn_timeout_seconds=turn_timeout
            )
            
            conversation_state = ConversationState(
                topic=topic,
                protocol_params=protocol_params
            )
            
            # Setup TTS if enabled
            tts_generator = None
            if enable_tts:
                tts_config = TTSConfig(
                    agent_1_voice=agent1_voice,
                    agent_2_voice=agent2_voice,
                    sample_rate=24000,
                    enable_realtime=True
                )
                tts_generator = DebateTTSGenerator(
                    config=tts_config,
                    output_dir="./audio_outputs"
                )
                current_tts_generator = tts_generator
            
            # Create protocol without event callback (we update history manually)
            protocol = ProtocolBuilder.create_custom_protocol(
                agent_1=agent_1,
                agent_2=agent_2,
                conversation_state=conversation_state,
                event_callback=None,  # We handle updates manually for real-time display
                tts_generator=tts_generator
            )
            
            current_protocol = protocol
            
            yield self._format_chat([], topic=topic), f"🎭 Starting debate on: {topic}", None
            
            # Run debate turn by turn with streaming
            turn = 0
            self.conversation_history = []
            
            while not conversation_state.is_terminated() and turn < max_turns:
                # Get current agent
                current_agent = protocol.agents_map[conversation_state.next_speaker]
                agent_label = "Agent 1" if conversation_state.next_speaker == "agent_1" else "Agent 2"
                
                # Update status - agent is thinking
                yield (
                    self._format_chat(self.conversation_history, topic=topic),
                    f"🎤 {agent_label} is thinking... (Turn {turn + 1}/{max_turns})",
                    None
                )
                
                # Execute turn
                try:
                    # Get context
                    context = conversation_state.get_context_for_agent(
                        conversation_state.next_speaker
                    )
                    
                    # Generate utterance with agent-specific temperature
                    agent_temp = agent1_temperature if current_agent.agent_id == "agent_1" else agent2_temperature
                    
                    # Start time for performance tracking
                    import time
                    start_time = time.time()
                    
                    utterance = current_agent.generate_utterance(
                        context=context,
                        temperature=agent_temp,
                        model=model_name,
                        max_tokens=max_tokens
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Append to state
                    conversation_state.append_utterance(
                        current_agent.agent_id,
                        utterance
                    )
                    
                    # Update conversation history immediately
                    self.conversation_history.append({
                        'agent_id': current_agent.agent_id,
                        'utterance': utterance,
                        'turn_number': conversation_state.current_turn_number
                    })
                    
                    # Yield immediately to show the new turn
                    status_msg = f"✓ {agent_label} responded (Turn {turn + 1}/{max_turns}, {generation_time:.1f}s)"
                    if enable_tts:
                        status_msg += " - Generating audio..."
                    
                    yield (
                        self._format_chat(self.conversation_history, topic=topic),
                        status_msg,
                        None
                    )
                    
                    # Generate TTS if enabled (after displaying text)
                    if enable_tts and tts_generator:
                        tts_start = time.time()
                        tts_generator.add_utterance(
                            text=utterance,
                            agent_id=current_agent.agent_id,
                            turn_number=conversation_state.current_turn_number,
                            add_pause=True
                        )
                        tts_time = time.time() - tts_start
                        
                        # Update with TTS completion
                        yield (
                            self._format_chat(self.conversation_history, topic=topic),
                            f"✓ Turn {turn + 1} completed (Gen: {generation_time:.1f}s, TTS: {tts_time:.1f}s)",
                            None
                        )
                    
                    turn += 1
                    
                except Exception as e:
                    logger.error(f"Error in turn {turn}: {e}", exc_info=True)
                    yield (
                        self._format_chat(self.conversation_history, topic=topic),
                        f"❌ Error in turn {turn + 1}: {str(e)}",
                        None
                    )
                    break
            
            # Debate completed
            metrics = conversation_state.compute_metrics()
            final_status = f"""✅ Debate Completed!
            
📊 Summary:
• Total Turns: {turn}
• Duration: {metrics.conversation_duration_seconds:.1f}s
• Avg Response Time: {metrics.conversation_duration_seconds / max(turn, 1):.1f}s per turn
• Agent 1: {metrics.agent_1_turns} turns
• Agent 2: {metrics.agent_2_turns} turns"""
            
            # Save and return audio if enabled
            audio_path = None
            if enable_tts and tts_generator:
                try:
                    yield (
                        self._format_chat(self.conversation_history, topic=topic),
                        final_status + "\n\n🎵 Generating final audio file...",
                        None
                    )
                    
                    audio_path = tts_generator.save_debate_audio(topic=topic)
                    stats = tts_generator.get_statistics()
                    final_status += f"\n\n🎵 Audio Ready:\n• Duration: {stats['total_duration']:.1f}s\n• File: {audio_path}"
                except Exception as e:
                    logger.error(f"Error saving audio: {e}", exc_info=True)
                    final_status += f"\n\n⚠️ Audio generation failed: {str(e)}"
            
            yield (
                self._format_chat(self.conversation_history, topic=topic),
                final_status,
                audio_path
            )
            
        except Exception as e:
            logger.error(f"Debate error: {e}", exc_info=True)
            yield (
                self._format_chat(self.conversation_history, topic=topic if 'topic' in locals() else None),
                f"❌ Error: {str(e)}",
                None
            )
    
    def _format_chat(self, history: list, topic: str = None) -> str:
        """Format conversation history as HTML."""
        if not history:
            if topic:
                return f"""
                <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 12px; margin-bottom: 20px;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h2 style='margin: 0; color: white; font-size: 1.3em; font-weight: 600;'>
                            🎭 Debate Topic
                        </h2>
                        <p style='margin: 10px 0 0 0; color: #e0e7ff; font-size: 1.1em; line-height: 1.5;'>
                            {topic}
                        </p>
                    </div>
                    <div style='text-align: center; padding: 40px; color: #999;'>
                        <p style='font-size: 1.1em;'>⏳ Waiting for agents to begin...</p>
                    </div>
                </div>
                """
            return """
            <div style='text-align: center; padding: 40px; color: #666; 
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
                <h3 style='margin: 0; font-size: 1.5em; color: #333;'>🎭 Ready to Debate</h3>
                <p style='margin: 10px 0 0 0; color: #999;'>Configure settings and click "Start Debate" to begin</p>
            </div>
            """
        
        html_parts = []
        
        # Add topic header if provided
        if topic:
            html_parts.append(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 12px; margin-bottom: 20px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='margin: 0; color: white; font-size: 1.3em; font-weight: 600;'>
                    🎭 Debate Topic
                </h2>
                <p style='margin: 10px 0 0 0; color: #e0e7ff; font-size: 1.1em; line-height: 1.5;'>
                    {topic}
                </p>
            </div>
            """)
        
        html_parts.append("""
        <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
                    max-height: 600px; overflow-y: auto; padding: 10px;' id='debate-container'>
        """)
        
        for msg in history:
            agent_num = "1" if msg['agent_id'] == 'agent_1' else "2"
            
            # Different colors for each agent
            if agent_num == "1":
                agent_color = "#2563eb"  # Blue
                bg_color = "#eff6ff"     # Light blue
                border_color = "#3b82f6"
                agent_emoji = "🔵"
            else:
                agent_color = "#059669"  # Green
                bg_color = "#f0fdf4"     # Light green
                border_color = "#10b981"
                agent_emoji = "🟢"
            
            html_parts.append(f"""
            <div style='margin: 15px 0; padding: 16px; background: {bg_color}; 
                        border-left: 4px solid {border_color}; border-radius: 8px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        animation: slideIn 0.3s ease-out;'>
                <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                    <span style='font-size: 1.2em; margin-right: 8px;'>{agent_emoji}</span>
                    <span style='font-weight: 600; color: {agent_color}; font-size: 0.95em;'>
                        Agent {agent_num}
                    </span>
                    <span style='margin-left: auto; color: #6b7280; font-size: 0.85em;'>
                        Turn {msg['turn_number']}
                    </span>
                </div>
                <div style='color: #1f2937; line-height: 1.7; font-size: 0.95em;'>
                    {msg['utterance']}
                </div>
            </div>
            """)
        
        html_parts.append("""
        </div>
        <style>
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            #debate-container::-webkit-scrollbar {
                width: 8px;
            }
            #debate-container::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            #debate-container::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 10px;
            }
            #debate-container::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        </style>
        <script>
            // Auto-scroll to bottom on new message
            const container = document.getElementById('debate-container');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        </script>
        """)
        
        return "".join(html_parts)


def create_gradio_interface():
    """Create the Gradio interface."""
    
    # Get API key from environment
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    # Initialize runner
    runner = DebateRunner(api_key)
    
    # Available models
    GROQ_MODELS = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama-3.3-70b-versatile",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    # Available voices
    FEMALE_VOICES = ["af_bella", "af_sarah", "af_nicole", "af_sky", "af_heart"]
    MALE_VOICES = ["am_adam", "am_michael", "am_eric"]
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .debate-display {
        max-height: 600px;
        overflow-y: auto;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Infinite Debate System") as app:
        gr.Markdown("""
        # 🎭 Infinite Debate System
        
        An AI-powered debate platform where two autonomous agents engage in structured discussions.
        Configure agents, choose voices, and watch them debate in real-time.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Debate display
                chat_display = gr.HTML(
                    value=runner._format_chat([]),
                    label="Debate",
                    elem_classes=["debate-display"]
                )
                
                status_display = gr.Textbox(
                    label="Status",
                    value="Ready to start",
                    interactive=False,
                    lines=2
                )
                
                audio_output = gr.Audio(
                    label="Debate Audio (if enabled)",
                    visible=True,
                    type="filepath"
                )
            
            with gr.Column(scale=1):
                with gr.Accordion("📝 Basic Settings", open=True):
                    topic_input = gr.Textbox(
                        label="Debate Topic",
                        placeholder="Enter the topic for debate...",
                        value="Would you go back in time and kill baby Hitler?",
                        lines=2
                    )
                    
                    max_turns = gr.Slider(
                        label="Maximum Turns",
                        minimum=2,
                        maximum=20,
                        value=6,
                        step=1
                    )
                    
                    model_select = gr.Dropdown(
                        label="LLM Model",
                        choices=GROQ_MODELS,
                        value="llama-3.1-8b-instant"
                    )
                    
                    max_tokens = gr.Slider(
                        label="Max Tokens per Response",
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=128
                    )
                
                with gr.Accordion("🤖 Agent 1 Configuration", open=False):
                    agent1_stance = gr.Textbox(
                        label="Stance/Position",
                        placeholder="Agent 1's position on the topic...",
                        value="I argue FOR the affirmative position",
                        lines=2
                    )
                    
                    agent1_personality = gr.Textbox(
                        label="Personality",
                        placeholder="Describe agent's personality...",
                        value="Analytical and evidence-focused",
                        lines=2
                    )
                    
                    agent1_temperature = gr.Slider(
                        label="Temperature (creativity)",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.9,
                        step=0.1
                    )
                    
                    agent1_system_prompt = gr.TextArea(
                        label="Custom System Prompt (optional)",
                        placeholder="Leave empty for default...",
                        lines=3
                    )
                
                with gr.Accordion("🤖 Agent 2 Configuration", open=False):
                    agent2_stance = gr.Textbox(
                        label="Stance/Position",
                        placeholder="Agent 2's position on the topic...",
                        value="I argue AGAINST the affirmative position",
                        lines=2
                    )
                    
                    agent2_personality = gr.Textbox(
                        label="Personality",
                        placeholder="Describe agent's personality...",
                        value="Philosophical and principle-based",
                        lines=2
                    )
                    
                    agent2_temperature = gr.Slider(
                        label="Temperature (creativity)",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.9,
                        step=0.1
                    )
                    
                    agent2_system_prompt = gr.TextArea(
                        label="Custom System Prompt (optional)",
                        placeholder="Leave empty for default...",
                        lines=3
                    )
                
                with gr.Accordion("🎤 TTS Configuration", open=False):
                    enable_tts = gr.Checkbox(
                        label="Enable Text-to-Speech Audio",
                        value=False
                    )
                    
                    agent1_voice = gr.Dropdown(
                        label="Agent 1 Voice",
                        choices=FEMALE_VOICES + MALE_VOICES,
                        value="af_bella"
                    )
                    
                    agent2_voice = gr.Dropdown(
                        label="Agent 2 Voice",
                        choices=FEMALE_VOICES + MALE_VOICES,
                        value="am_adam"
                    )
                    
                    gr.Markdown("""
                    **Available Voices:**
                    - Female: bella, sarah, nicole, sky, heart
                    - Male: adam, michael, eric
                    """)
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    turn_timeout = gr.Slider(
                        label="Turn Timeout (seconds)",
                        minimum=10,
                        maximum=120,
                        value=30,
                        step=5
                    )
                    
                    gr.Markdown("""
                    **Note:** Advanced features like latency optimization 
                    and token/sec metrics are displayed in logs.
                    """)
                
                with gr.Row():
                    start_btn = gr.Button("🎬 Start Debate", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                
                with gr.Accordion("ℹ️ Quick Presets", open=False):
                    gr.Markdown("""
                    **Philosophical Debate:**
                    - Topic: "Is free will an illusion?"
                    - Agent 1: "Free will is an illusion" (af_sarah)
                    - Agent 2: "Free will is real" (am_michael)
                    
                    **Political Debate:**
                    - Topic: "Should AI be regulated?"
                    - Agent 1: "Yes, regulation needed" (af_bella)
                    - Agent 2: "No, innovation first" (am_adam)
                    
                    **Ethical Debate:**
                    - Topic: "Is eating meat ethical?"
                    - Agent 1: "No, it's unethical" (af_nicole)
                    - Agent 2: "Yes, it's natural" (am_eric)
                    """)
        
        # Event handlers
        start_btn.click(
            fn=runner.run_debate_streaming,
            inputs=[
                # Basic
                topic_input,
                max_turns,
                # Agent 1
                agent1_stance,
                agent1_personality,
                agent1_temperature,
                agent1_system_prompt,
                # Agent 2
                agent2_stance,
                agent2_personality,
                agent2_temperature,
                agent2_system_prompt,
                # Model
                model_select,
                max_tokens,
                # TTS
                enable_tts,
                agent1_voice,
                agent2_voice,
                # Advanced
                turn_timeout
            ],
            outputs=[chat_display, status_display, audio_output]
        )
        
        gr.Markdown("""
        ---
        
        ### 📚 Usage Tips
        
        1. **Quick Start:** Use default settings and just enter a topic
        2. **Custom Agents:** Expand agent accordions to customize personalities
        3. **Audio:** Enable TTS to hear the debate (requires espeak-ng)
        4. **Models:** Try different models for varied debate styles
        
        ### 🔧 Setup
        
        Make sure you have:
        - GROQ_API_KEY set in environment
        - espeak-ng installed (for TTS)
        - All dependencies from requirements.txt
        
        ### 📖 Examples
        
        - **Classic:** "Would you go back in time and kill baby Hitler?"
        - **Tech:** "Will AI replace human jobs?"
        - **Philosophy:** "Is consciousness purely physical?"
        - **Ethics:** "Should we colonize Mars?"
        """)
    
    return app


def main():
    """Launch the Gradio app."""
    
    # Check for API key
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not found in environment")
        print("\n⚠️  WARNING: GROQ_API_KEY not set!")
        print("Please set it in your .env file or environment variables")
        print("The app will start but debates will fail without an API key\n")
    
    # Create and launch app
    app = create_gradio_interface()
    
    logger.info("Launching Gradio interface...")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
