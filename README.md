# Logorrhea: Multi Agent AI Debate System

This is a multi-agent conversational framework that orchestrates autonomous debates between AI agents.

## Philosophy

I wanted to see how well they can argue, how well they preserve their statements (no bullshit like, 'you're absolutely right!'). We ask them questions, they respond, and the conversation ends. But what happens when we remove ourselves from the loop entirely? The goal of this project is to understand what emerges when two AI agents are given a contentious topic and told simply: debate, indefinitely?

#### Some questions i want to see being answered:
- How arguments evolve and deteriorate over long conversations?
- Whether novel insights emerge from extended dialectics?
- Hallucinations: Do they converge toward truth, or drift into some absurdity?
- How positions shift (or don't) without external validation?
- At what point do arguments become circular?

#### My policy while making this project is to:
- have cost $0. use entirely open source models (mostly from hf) (rn, it uses free api credits for llm call)
- use synthetic voices, dont try to impersonate a specific human (debater/philopher)




## Features

- **Stateless Agents**: Pure functions that read state and generate responses
- **Immutable Transcript**: Conversation history cannot be retroactively altered
- **Explicit Protocol Control**: Turn-taking governed at the system level, not by agents
- **LangGraph Integration**: Observable, structured workflow execution
- **Real-time TTS**: Kokoro-powered voice synthesis with 8 distinct voices
- **Single Audio Output**: Entire debates combined into one continuous WAV file


## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/nihilisticneuralnet/Logorrhea.git
cd Logorrhea

# Run setup script (handles everything)
chmod +x setup.sh
./setup.sh

# Or manual setup:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install espeak-ng for TTS (required)
# Linux:
sudo apt-get install espeak-ng
# macOS:
brew install espeak-ng

# Configure API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```


## Quick Start

### Basic Debate

```python
from main import InfiniteDebateSystem

system = InfiniteDebateSystem(api_key="your_groq_api_key")

final_state = system.run_debate_with_protocol(
    topic="Is free will an illusion?",
    max_turns=10
)

system.display_conversation(final_state)
```

### With Voice Synthesis

```python
from main import InfiniteDebateSystem
from tts import TTSConfig

system = InfiniteDebateSystem(api_key="your_groq_api_key")

tts_config = TTSConfig(
    agent_1_voice="af_bella",  # Female voice
    agent_2_voice="am_adam"     # Male voice
)

final_state = system.run_debate_with_protocol(
    topic="Would you go back in time and kill baby Hitler?",
    max_turns=8,
    enable_tts=True,
    tts_config=tts_config
)

# Audio saved to: ./audio_outputs/debate_[topic].wav
```

## Configuration

### Protocol Parameters

```python
from conversation_state import ProtocolParameters

params = ProtocolParameters(
    max_turns=None,              # None = infinite (limited by external factors)
    max_utterance_length=2000,   # Characters per response
    temperature=1.0,             # LLM creativity (0.0-2.0)
    model="llama-3.1-8b-instant", # Groq model
    turn_timeout_seconds=30      # Max time per turn
)
```

### Agent Customization

```python
from agents import AgentFactory

agent_1, agent_2 = AgentFactory.create_debate_pair(
    api_key=api_key,
    topic="Your topic",
    agent_1_stance="Specific position to argue",
    agent_2_stance="Opposing position to argue",
    agent_1_personality="Analytical and data-driven",
    agent_2_personality="Philosophical and principle-based"
)
```

### TTS Configuration

```python
from tts import TTSConfig

tts_config = TTSConfig(
    agent_1_voice="af_bella",
    agent_2_voice="am_adam",
    sample_rate=24000,       # Audio quality
    enable_realtime=True     # Generate during debate
)
```

## Future

- [] debate between more than 2 people
- [] user can control variables (custom prompt, temp, small vs big params)
- [] Introduce an external “environment token” that changes slowly (like temperature or season)
- [] user can interrupt in bt (idk why might delte this)
- [] never eneding conversation util paused (memory handle)


# TO DO
- [] CHECK LATENCY (due to open spurce models)
- [] BUILD MEMORY HANDLING
- [] FROTEND FOR CHAT OR ATLEAST LOGS OF CHATS

## References:
- https://jamez.it/project/the-infinite-conversation/

### TTS
- https://huggingface.co/hexgrad/Kokoro-82M
- https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS
