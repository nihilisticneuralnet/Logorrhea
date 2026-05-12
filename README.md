# Logorrhea: Multi Agent AI Debate System

A multi-agent conversational framework that orchestrates autonomous debates between AI agents, with a hierarchical memory architecture.

## Philosophy

Some questions I like to see being answered:
- How arguments evolve and deteriorate over long conversations?
- Whether novel insights emerge from extended dialectics?
- Do arguments become circular, do they converge towards a point, or shift into some absurdity?
- How positions shift (or don't) without external validation?


"""
Five-layer memory model:
  L1  Short-Term Working Memory   — verbatim recent turns (ring buffer)
  L2  Episodic Memory             — structured semantic events / claim graph
  L3  Semantic Memory             — cross-conversation factual knowledge
  L4  Agent-Private Memory        — beliefs, ideology, strategy, trust
  L5  Global Shared Memory        — world state, shared facts, debate history


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


## Installation

### Setup

```bash
# Clone the repository
git clone https://github.com/nihilisticneuralnet/Logorrhea.git
cd Logorrhea

# Install dependencies
pip install -r requirements.txt

# Insert your API keys
export GROQ_API_KEY="your_groq_api_key_here" # or hf_token (any one)

# Run tests
cd src
python main.py

# For gradio interface, use
python app.py
```


## Example 
Topic: Would you go back in time and kill baby Hitler?


https://github.com/user-attachments/assets/a88e1270-bbd9-4245-840c-c2eef44f189c



## References
- https://jamez.it/project/the-infinite-conversation/
- https://huggingface.co/hexgrad/Kokoro-82M
- https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS
