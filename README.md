# Logorrhea: Multi Agent AI Debate System

A multi-agent conversational framework that orchestrates autonomous debates between AI agents, with a five-layer hierarchical memory architecture.

## Philosophy

Some questions I want to see being answered:
- How arguments evolve and deteriorate over long conversations?
- Whether novel insights emerge from extended dialectics?
- Hallucinations: Do they converge towards a point, or shift into some absurdity?
- How positions shift (or don't) without external validation?
- At what point do arguments become circular?




"""
Five-layer memory model:
  L1  Short-Term Working Memory   — verbatim recent turns (ring buffer)
  L2  Episodic Memory             — structured semantic events / claim graph
  L3  Semantic Memory             — cross-conversation factual knowledge
  L4  Agent-Private Memory        — beliefs, ideology, strategy, trust
  L5  Global Shared Memory        — world state, shared facts, debate history

Storage backend: SQLite + LanceDB (zero-dependency, embeddable, pgvector-compatible
interface so swapping to PostgreSQL+pgvector requires only changing the
DatabaseBackend implementation).

Key design choices
------------------
* Event-boundary compression  — summaries are triggered by semantic shifts
  (topic change, contradiction, resolution) NOT by fixed turn windows.
* Hybrid retrieval             — vector + keyword + temporal + graph traversal.
* Claim graph                  — nodes=claims/evidence/hypotheses,
                                 edges=support/contradict/derive/consensus.
* Provenance on every object   — speaker, turns, timestamps, confidence,
                                 retrieval_count, conversation_lineage.
* Semantic deduplication       — cosine-similarity gate before insert.
* Memory decay                 — exponential staleness penalty on retrieval score.
* Stochastic / novelty / contradiction retrieval — breaks feedback loops.
"""

## Architecture

  <img src="Logorrhea.png" />

## Features

- **Stateless Agents**: Pure functions that read state and generate responses
- **Explicit Protocol Control**: Turn-taking governed at the system level, not by agents
- **Real-time TTS**: Kokoro-powered voice synthesis with 8 distinct voices


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




## References:
- https://jamez.it/project/the-infinite-conversation/
- https://huggingface.co/hexgrad/Kokoro-82M
- https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS
