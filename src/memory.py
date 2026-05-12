from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    logger.warning("numpy not found — cosine similarity will use pure Python")

try:
    import lancedb
    _HAS_LANCEDB = True
except ImportError:
    _HAS_LANCEDB = False
    logger.warning("lancedb not found — falling back to SQLite float-array vectors")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False
    logger.warning(
        "sentence-transformers not found — using hash-based pseudo-embeddings. "
        "Install with: pip install sentence-transformers"
    )


class MemoryLayer(str, Enum):
    SHORT_TERM  = "short_term"
    EPISODIC    = "episodic"
    SEMANTIC    = "semantic"
    AGENT       = "agent"
    GLOBAL      = "global"


class ClaimRelation(str, Enum):
    SUPPORT      = "support"
    CONTRADICT   = "contradict"
    DERIVE       = "derive"
    CONSENSUS    = "consensus"
    REFINE       = "refine"
    TEMPORAL     = "temporal"
    UNCERTAINTY  = "uncertainty"


class EpisodeEventType(str, Enum):
    CLAIM          = "claim"
    COUNTER_CLAIM  = "counter_claim"
    EVIDENCE       = "evidence"
    RESOLUTION     = "resolution"
    TOPIC_SHIFT    = "topic_shift"
    CONTRADICTION  = "contradiction"
    AGREEMENT      = "agreement"
    QUESTION       = "question"
    EMOTIONAL      = "emotional"
    STRATEGIC      = "strategic"
    UNRESOLVED     = "unresolved"


DEDUP_THRESHOLD = 0.88 # Deduplication similarity threshold (0-1)
DEFAULT_TOKEN_BUDGET = 1_200 # Max tokens allocated for memory context passed to agent
SHORT_TERM_WINDOW = 12 # Short-term ring buffer size (turns)
DECAY_HALF_LIFE = 7 * 24 * 3600 # Decay half-life in seconds (7 days)
PSEUDO_DIM = 128 # Embedding dimension for the pseudo-embedder fallback


class Embedder:
    """
    Thin wrapper around sentence-transformers with a hash-based fallback
    so the system works even without the ML library installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = None
        self._dim = PSEUDO_DIM
        if _HAS_EMBEDDINGS:
            try:
                self._model = SentenceTransformer(model_name)
                self._dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedder: loaded '{model_name}' (dim={self._dim})")
            except Exception as exc:
                logger.warning(f"Could not load SentenceTransformer: {exc}")

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        return self._pseudo_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]
        return [self._pseudo_embed(t) for t in texts]

    def _pseudo_embed(self, text: str) -> List[float]:
        """Deterministic pseudo-embedding via hashing (for fallback)."""
        seed_bytes = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(0, PSEUDO_DIM * 4, 4):
            b = seed_bytes[i % 32 : (i % 32) + 4]
            val = int.from_bytes(b, "little") / 2**31 - 1.0
            vec.append(val)
        # L2-normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@dataclass
class Provenance:
    speaker_id: str
    turn_index: int
    timestamp: float = field(default_factory=lambda: time.time())
    conversation_id: str = ""
    confidence: float = 1.0
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None
    lineage: List[str] = field(default_factory=list)   # parent memory IDs


@dataclass
class MemoryObject:
    memory_id:   str
    layer:       MemoryLayer
    content:     str                      # human-readable text
    embedding:   List[float]
    provenance:  Provenance
    metadata:    Dict[str, Any] = field(default_factory=dict)
    keywords:    List[str] = field(default_factory=list)
    created_at:  float = field(default_factory=lambda: time.time())

    def staleness_penalty(self) -> float:
        """Exponential decay based on age."""
        age = time.time() - self.created_at
        return math.exp(-0.693 * age / DECAY_HALF_LIFE)   # 0.693 ≈ ln(2)


@dataclass
class ClaimNode:
    node_id:   str
    text:      str
    node_type: str                       # claim / evidence / hypothesis / decision
    speaker:   str
    turn:      int
    embedding: List[float]
    conv_id:   str
    created_at: float = field(default_factory=lambda: time.time())
    confidence: float = 1.0
    metadata:  Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimEdge:
    edge_id:   str
    src:       str                       # node_id
    dst:       str                       # node_id
    relation:  ClaimRelation
    weight:    float = 1.0
    created_at: float = field(default_factory=lambda: time.time())
    metadata:  Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeEvent:
    event_id:   str
    event_type: EpisodeEventType
    summary:    str
    embedding:  List[float]
    provenance: Provenance
    claim_ids:  List[str] = field(default_factory=list)
    entities:   List[str] = field(default_factory=list)
    resolved:   bool = False
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentBelief:
    belief_id:   str
    agent_id:    str
    content:     str
    belief_type: str         # ideological / strategic / trust / goal / hidden
    embedding:   List[float]
    confidence:  float = 1.0
    updated_at:  float = field(default_factory=lambda: time.time())
    conv_id:     str = ""
    metadata:    Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedContext:
    """Assembled context passed to the agent before each turn."""
    short_term_turns:    List[Dict]        # raw recent turns
    active_claims:       List[ClaimNode]
    unresolved_episodes: List[EpisodeEvent]
    agent_beliefs:       List[AgentBelief]
    semantic_facts:      List[MemoryObject]
    global_facts:        List[MemoryObject]
    token_estimate:      int = 0

    def to_prompt_block(self) -> str:
        parts: List[str] = []
        if self.short_term_turns:
            parts.append("### Recent Turns ###")
            for t in self.short_term_turns:
                parts.append(f"[{t['speaker']}]: {t['text']}")

        if self.active_claims:
            parts.append("\n### Active Claims in Debate ###")
            for c in self.active_claims:
                parts.append(f"- [{c.node_type}] ({c.speaker}): {c.text}")

        if self.unresolved_episodes:
            parts.append("\n### Unresolved Issues ###")
            for e in self.unresolved_episodes:
                parts.append(f"- [{e.event_type.value}]: {e.summary}")

        if self.agent_beliefs:
            parts.append("\n### Your Persistent Beliefs ###")
            for b in self.agent_beliefs:
                parts.append(f"- [{b.belief_type}] {b.content}")

        if self.semantic_facts:
            parts.append("\n### Relevant Background Knowledge ###")
            for m in self.semantic_facts:
                parts.append(f"- {m.content}")

        if self.global_facts:
            parts.append("\n### Shared World State ###")
            for m in self.global_facts:
                parts.append(f"- {m.content}")

        return "\n".join(parts)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if _HAS_NUMPY:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) or 1.0
    nb  = math.sqrt(x * x for x in b) if False else math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _rough_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token)."""
    return max(1, len(text) // 4)


def _extract_keywords(text: str) -> List[str]:
    """Naive keyword extraction — replace with YAKE/KeyBERT if available."""
    stopwords = {
        "the","a","an","is","was","are","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","that","this","these",
        "those","i","we","you","he","she","it","they","and","or",
        "but","in","on","at","to","for","of","with","by","from",
        "up","about","into","through","during","before","after",
        "above","below","between","out","off","over","under","again",
    }
    words = text.lower().split()
    kws = [w.strip(".,!?;:\"'()") for w in words if w not in stopwords and len(w) > 3]
    return list(dict.fromkeys(kws))[:20]


class DatabaseBackend:
    """
    SQLite-backed persistent store with optional LanceDB vector index.
    Designed for easy swap to PostgreSQL+pgvector:
      — replace _conn() with psycopg2/asyncpg
      — replace vector storage with pgvector extension
    """

    def __init__(self, db_path: str = "./memory_store/debate_memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._init_schema()
        logger.info(f"DatabaseBackend initialised at {db_path}")

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self._path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self):
        with self._conn() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS memory_objects (
                memory_id    TEXT PRIMARY KEY,
                layer        TEXT NOT NULL,
                content      TEXT NOT NULL,
                embedding    TEXT NOT NULL,   -- JSON array
                provenance   TEXT NOT NULL,   -- JSON
                metadata     TEXT DEFAULT '{}',
                keywords     TEXT DEFAULT '[]',
                created_at   REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claim_nodes (
                node_id     TEXT PRIMARY KEY,
                text        TEXT NOT NULL,
                node_type   TEXT NOT NULL,
                speaker     TEXT NOT NULL,
                turn_idx    INTEGER NOT NULL,
                embedding   TEXT NOT NULL,
                conv_id     TEXT NOT NULL,
                created_at  REAL NOT NULL,
                confidence  REAL DEFAULT 1.0,
                metadata    TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS claim_edges (
                edge_id     TEXT PRIMARY KEY,
                src         TEXT NOT NULL,
                dst         TEXT NOT NULL,
                relation    TEXT NOT NULL,
                weight      REAL DEFAULT 1.0,
                created_at  REAL NOT NULL,
                metadata    TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS episode_events (
                event_id    TEXT PRIMARY KEY,
                event_type  TEXT NOT NULL,
                summary     TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                provenance  TEXT NOT NULL,
                claim_ids   TEXT DEFAULT '[]',
                entities    TEXT DEFAULT '[]',
                resolved    INTEGER DEFAULT 0,
                metadata    TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS agent_beliefs (
                belief_id    TEXT PRIMARY KEY,
                agent_id     TEXT NOT NULL,
                content      TEXT NOT NULL,
                belief_type  TEXT NOT NULL,
                embedding    TEXT NOT NULL,
                confidence   REAL DEFAULT 1.0,
                updated_at   REAL NOT NULL,
                conv_id      TEXT DEFAULT '',
                metadata     TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS short_term_turns (
                turn_id      TEXT PRIMARY KEY,
                conv_id      TEXT NOT NULL,
                speaker      TEXT NOT NULL,
                text         TEXT NOT NULL,
                turn_index   INTEGER NOT NULL,
                created_at   REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_mo_layer     ON memory_objects(layer);
            CREATE INDEX IF NOT EXISTS idx_cn_conv      ON claim_nodes(conv_id);
            CREATE INDEX IF NOT EXISTS idx_ce_src       ON claim_edges(src);
            CREATE INDEX IF NOT EXISTS idx_ce_dst       ON claim_edges(dst);
            CREATE INDEX IF NOT EXISTS idx_ee_resolved  ON episode_events(resolved);
            CREATE INDEX IF NOT EXISTS idx_ab_agent     ON agent_beliefs(agent_id);
            CREATE INDEX IF NOT EXISTS idx_st_conv      ON short_term_turns(conv_id);
            """)

    def upsert_memory(self, obj: MemoryObject):
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO memory_objects
                    (memory_id, layer, content, embedding, provenance,
                     metadata, keywords, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                obj.memory_id, obj.layer.value, obj.content,
                json.dumps(obj.embedding), json.dumps(asdict(obj.provenance)),
                json.dumps(obj.metadata), json.dumps(obj.keywords),
                obj.created_at,
            ))

    def fetch_memories_by_layer(
        self, layer: MemoryLayer, limit: int = 200
    ) -> List[MemoryObject]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM memory_objects WHERE layer=? ORDER BY created_at DESC LIMIT ?",
                (layer.value, limit)
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def increment_retrieval(self, memory_id: str):
        with self._conn() as con:
            con.execute("""
                UPDATE memory_objects
                SET provenance = json_set(provenance,
                    '$.retrieval_count', CAST(json_extract(provenance,'$.retrieval_count') AS INT)+1,
                    '$.last_retrieved', ?)
                WHERE memory_id=?
            """, (time.time(), memory_id))

    def _row_to_memory(self, r) -> MemoryObject:
        prov_d = json.loads(r["provenance"])
        prov   = Provenance(**prov_d)
        return MemoryObject(
            memory_id  = r["memory_id"],
            layer      = MemoryLayer(r["layer"]),
            content    = r["content"],
            embedding  = json.loads(r["embedding"]),
            provenance = prov,
            metadata   = json.loads(r["metadata"]),
            keywords   = json.loads(r["keywords"]),
            created_at = r["created_at"],
        )

    def upsert_claim_node(self, node: ClaimNode):
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO claim_nodes
                    (node_id, text, node_type, speaker, turn_idx,
                     embedding, conv_id, created_at, confidence, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                node.node_id, node.text, node.node_type, node.speaker,
                node.turn, json.dumps(node.embedding), node.conv_id,
                node.created_at, node.confidence, json.dumps(node.metadata),
            ))

    def upsert_claim_edge(self, edge: ClaimEdge):
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO claim_edges
                    (edge_id, src, dst, relation, weight, created_at, metadata)
                VALUES (?,?,?,?,?,?,?)
            """, (
                edge.edge_id, edge.src, edge.dst, edge.relation.value,
                edge.weight, edge.created_at, json.dumps(edge.metadata),
            ))

    def get_claim_nodes(self, conv_id: str, limit: int = 50) -> List[ClaimNode]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM claim_nodes WHERE conv_id=? ORDER BY created_at DESC LIMIT ?",
                (conv_id, limit)
            ).fetchall()
        return [self._row_to_claim_node(r) for r in rows]

    def get_edges_for_node(self, node_id: str) -> List[ClaimEdge]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM claim_edges WHERE src=? OR dst=?",
                (node_id, node_id)
            ).fetchall()
        return [self._row_to_claim_edge(r) for r in rows]

    def _row_to_claim_node(self, r) -> ClaimNode:
        return ClaimNode(
            node_id=r["node_id"], text=r["text"], node_type=r["node_type"],
            speaker=r["speaker"], turn=r["turn_idx"],
            embedding=json.loads(r["embedding"]), conv_id=r["conv_id"],
            created_at=r["created_at"], confidence=r["confidence"],
            metadata=json.loads(r["metadata"]),
        )

    def _row_to_claim_edge(self, r) -> ClaimEdge:
        return ClaimEdge(
            edge_id=r["edge_id"], src=r["src"], dst=r["dst"],
            relation=ClaimRelation(r["relation"]),
            weight=r["weight"], created_at=r["created_at"],
            metadata=json.loads(r["metadata"]),
        )

    def upsert_episode(self, evt: EpisodeEvent):
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO episode_events
                    (event_id, event_type, summary, embedding, provenance,
                     claim_ids, entities, resolved, metadata)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                evt.event_id, evt.event_type.value, evt.summary,
                json.dumps(evt.embedding), json.dumps(asdict(evt.provenance)),
                json.dumps(evt.claim_ids), json.dumps(evt.entities),
                int(evt.resolved), json.dumps(evt.metadata),
            ))

    def get_unresolved_episodes(
        self, conv_id: str, limit: int = 10
    ) -> List[EpisodeEvent]:
        with self._conn() as con:
            rows = con.execute("""
                SELECT * FROM episode_events
                WHERE resolved=0
                  AND json_extract(provenance,'$.conversation_id')=?
                ORDER BY json_extract(provenance,'$.timestamp') DESC
                LIMIT ?
            """, (conv_id, limit)).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_all_episodes(self, conv_id: str) -> List[EpisodeEvent]:
        with self._conn() as con:
            rows = con.execute("""
                SELECT * FROM episode_events
                WHERE json_extract(provenance,'$.conversation_id')=?
                ORDER BY json_extract(provenance,'$.timestamp') DESC
            """, (conv_id,)).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def _row_to_episode(self, r) -> EpisodeEvent:
        prov_d = json.loads(r["provenance"])
        return EpisodeEvent(
            event_id=r["event_id"],
            event_type=EpisodeEventType(r["event_type"]),
            summary=r["summary"],
            embedding=json.loads(r["embedding"]),
            provenance=Provenance(**prov_d),
            claim_ids=json.loads(r["claim_ids"]),
            entities=json.loads(r["entities"]),
            resolved=bool(r["resolved"]),
            metadata=json.loads(r["metadata"]),
        )

    def upsert_belief(self, belief: AgentBelief):
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO agent_beliefs
                    (belief_id, agent_id, content, belief_type, embedding,
                     confidence, updated_at, conv_id, metadata)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                belief.belief_id, belief.agent_id, belief.content,
                belief.belief_type, json.dumps(belief.embedding),
                belief.confidence, belief.updated_at,
                belief.conv_id, json.dumps(belief.metadata),
            ))

    def get_beliefs(self, agent_id: str) -> List[AgentBelief]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM agent_beliefs WHERE agent_id=? ORDER BY updated_at DESC",
                (agent_id,)
            ).fetchall()
        return [self._row_to_belief(r) for r in rows]

    def _row_to_belief(self, r) -> AgentBelief:
        return AgentBelief(
            belief_id=r["belief_id"], agent_id=r["agent_id"],
            content=r["content"], belief_type=r["belief_type"],
            embedding=json.loads(r["embedding"]),
            confidence=r["confidence"], updated_at=r["updated_at"],
            conv_id=r["conv_id"], metadata=json.loads(r["metadata"]),
        )


    def push_turn(
        self, conv_id: str, speaker: str, text: str, turn_index: int
    ):
        with self._conn() as con:
            con.execute("""
                INSERT INTO short_term_turns
                    (turn_id, conv_id, speaker, text, turn_index, created_at)
                VALUES (?,?,?,?,?,?)
            """, (_new_id("turn"), conv_id, speaker, text, turn_index, time.time()))
            # Trim to ring-buffer window
            con.execute("""
                DELETE FROM short_term_turns
                WHERE conv_id=?
                  AND turn_id NOT IN (
                      SELECT turn_id FROM short_term_turns
                      WHERE conv_id=?
                      ORDER BY turn_index DESC
                      LIMIT ?
                  )
            """, (conv_id, conv_id, SHORT_TERM_WINDOW))

    def get_recent_turns(
        self, conv_id: str, n: int = SHORT_TERM_WINDOW
    ) -> List[Dict]:
        with self._conn() as con:
            rows = con.execute("""
                SELECT speaker, text, turn_index, created_at
                FROM short_term_turns
                WHERE conv_id=?
                ORDER BY turn_index DESC
                LIMIT ?
            """, (conv_id, n)).fetchall()
        turns = [dict(r) for r in rows]
        return list(reversed(turns))


class CompressionPipeline:
    """
    Transforms raw turn sequences into structured MemoryObjects and
    EpisodeEvents at semantic event boundaries, not fixed windows.
    """

    # Heuristic: keywords that signal a boundary event
    _BOUNDARY_SIGNALS = {
        EpisodeEventType.TOPIC_SHIFT:   ["however","turning to","new point","different angle","let me shift"],
        EpisodeEventType.CONTRADICTION: ["contradict","disagree","wrong","false","dispute","refute","challenge"],
        EpisodeEventType.RESOLUTION:    ["agree","concede","settled","conclude","consensus","granted","accept"],
        EpisodeEventType.UNRESOLVED:    ["unresolved","unclear","open question","still debating","further"],
        EpisodeEventType.EVIDENCE:      ["evidence","data","study","research","according to","statistics","proof"],
        EpisodeEventType.STRATEGIC:     ["strategy","reframe","deflect","pivot","rhetorical","trap"],
    }

    def __init__(self, embedder: Embedder):
        self._emb = embedder

    def detect_boundary(self, text: str) -> Optional[EpisodeEventType]:
        tl = text.lower()
        for etype, signals in self._BOUNDARY_SIGNALS.items():
            if any(s in tl for s in signals):
                return etype
        return None

    def extract_claims(
        self, text: str, speaker: str, turn: int, conv_id: str
    ) -> List[ClaimNode]:
        """
        Naive sentence-level claim extractor.
        Replace with an LLM tool-call extraction step for production.
        """
        sentences = [s.strip() for s in text.replace("?", ".").split(".") if len(s.strip()) > 20]
        nodes = []
        for sent in sentences[:4]:        # cap at 4 claims per turn
            ntype = "claim"
            sl = sent.lower()
            if any(w in sl for w in ["evidence","data","study","research","proof"]):
                ntype = "evidence"
            elif sent.strip().endswith("?"):
                ntype = "question"
            elif any(w in sl for w in ["hypothesis","suppose","assume","if we"]):
                ntype = "hypothesis"
            nodes.append(ClaimNode(
                node_id    = _new_id("cn"),
                text       = sent,
                node_type  = ntype,
                speaker    = speaker,
                turn       = turn,
                embedding  = self._emb.embed(sent),
                conv_id    = conv_id,
                confidence = 0.9,
            ))
        return nodes

    def create_episode_event(
        self,
        event_type: EpisodeEventType,
        turns_window: List[Dict],
        claim_ids: List[str],
        speaker: str,
        turn: int,
        conv_id: str,
    ) -> EpisodeEvent:
        summary_text = " | ".join(t["text"][:120] for t in turns_window[-3:])
        summary = f"[{event_type.value.upper()}] {summary_text}"
        entities = list({t["speaker"] for t in turns_window})
        return EpisodeEvent(
            event_id   = _new_id("ep"),
            event_type = event_type,
            summary    = summary,
            embedding  = self._emb.embed(summary),
            provenance = Provenance(
                speaker_id      = speaker,
                turn_index      = turn,
                conversation_id = conv_id,
            ),
            claim_ids = claim_ids,
            entities  = entities,
        )

    def compress_to_semantic(
        self,
        episodes: List[EpisodeEvent],
        conv_id: str,
    ) -> MemoryObject:
        """Distil a cluster of episodes into a canonical semantic memory."""
        combined = "; ".join(e.summary[:100] for e in episodes)
        distilled = f"Debate insight: {combined}"
        return MemoryObject(
            memory_id  = _new_id("sem"),
            layer      = MemoryLayer.SEMANTIC,
            content    = distilled,
            embedding  = self._emb.embed(distilled),
            provenance = Provenance(
                speaker_id      = "system",
                turn_index      = -1,
                conversation_id = conv_id,
                confidence      = 0.8,
                lineage         = [e.event_id for e in episodes],
            ),
            keywords   = _extract_keywords(distilled),
        )


class HybridRetriever:
    """
    Combines:
      1. Vector similarity (cosine)
      2. Keyword overlap (BM25-lite)
      3. Temporal recency bonus
      4. Contradiction prioritisation
      5. Novelty weighting (penalise over-retrieved memories)
      6. Stochastic jitter (breaks feedback loops)
    """

    def __init__(self, embedder: Embedder):
        self._emb = embedder

    def score(
        self,
        query_vec: List[float],
        query_kws: List[str],
        mem: MemoryObject,
        *,
        stochastic: bool = True,
        novelty_weight: float = 0.15,
        contradiction_boost: bool = False,
    ) -> float:
        # 1. Vector similarity
        vec_score = _cosine_sim(query_vec, mem.embedding)

        # 2. Keyword overlap
        mem_kws = set(mem.keywords)
        kw_score = len(set(query_kws) & mem_kws) / (len(query_kws) + 1)

        # 3. Temporal recency
        recency = mem.staleness_penalty()

        # 4. Novelty — penalise over-retrieved
        retrieval_penalty = novelty_weight / (1 + mem.provenance.retrieval_count)

        # 5. Contradiction boost
        cboost = 0.1 if contradiction_boost and "contradict" in mem.keywords else 0.0

        score = (0.55 * vec_score + 0.20 * kw_score + 0.15 * recency
                 + retrieval_penalty + cboost)

        # 6. Stochastic jitter ±3 %
        if stochastic:
            score += random.uniform(-0.03, 0.03)

        return score

    def rank(
        self,
        query: str,
        candidates: List[MemoryObject],
        top_k: int = 5,
        **kwargs,
    ) -> List[MemoryObject]:
        if not candidates:
            return []
        q_vec = self._emb.embed(query)
        q_kws = _extract_keywords(query)
        scored = [(self.score(q_vec, q_kws, m, **kwargs), m) for m in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def rank_claims(
        self, query: str, candidates: List[ClaimNode], top_k: int = 6
    ) -> List[ClaimNode]:
        if not candidates:
            return []
        q_vec = self._emb.embed(query)
        scored = [(_cosine_sim(q_vec, c.embedding) + random.uniform(-0.02, 0.02), c)
                  for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def rank_episodes(
        self, query: str, candidates: List[EpisodeEvent], top_k: int = 5
    ) -> List[EpisodeEvent]:
        if not candidates:
            return []
        q_vec = self._emb.embed(query)
        scored = [(_cosine_sim(q_vec, e.embedding) + random.uniform(-0.02, 0.02), e)
                  for e in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]


class SemanticDeduplicator:
    """
    Prevents memory explosion from paraphrased restatements.
    Maintains an in-memory index of recent embeddings per layer
    for fast candidate comparison.
    """

    def __init__(self, threshold: float = DEDUP_THRESHOLD):
        self._threshold = threshold
        self._index: Dict[str, List[Tuple[str, List[float]]]] = {}

    def is_duplicate(self, layer: str, embedding: List[float]) -> bool:
        candidates = self._index.get(layer, [])
        for _, existing_emb in candidates[-200:]:   # check last 200 per layer
            if _cosine_sim(embedding, existing_emb) >= self._threshold:
                return True
        return False

    def register(self, layer: str, memory_id: str, embedding: List[float]):
        self._index.setdefault(layer, []).append((memory_id, embedding))

    def rebuild_from_db(self, db: DatabaseBackend):
        """Warm-up on restart from persisted objects."""
        for layer in MemoryLayer:
            mems = db.fetch_memories_by_layer(layer, limit=500)
            for m in mems:
                self._index.setdefault(layer.value, []).append(
                    (m.memory_id, m.embedding)
                )


class ClaimGraphManager:
    """
    Manages claim nodes and labelled edges.
    Provides graph-traversal retrieval for contradiction chains,
    support trees, and consensus paths.
    """

    def __init__(self, db: DatabaseBackend, embedder: Embedder):
        self._db  = db
        self._emb = embedder

    def add_claim(
        self,
        text: str,
        speaker: str,
        turn: int,
        conv_id: str,
        node_type: str = "claim",
        confidence: float = 0.9,
    ) -> ClaimNode:
        node = ClaimNode(
            node_id   = _new_id("cn"),
            text      = text,
            node_type = node_type,
            speaker   = speaker,
            turn      = turn,
            embedding = self._emb.embed(text),
            conv_id   = conv_id,
            confidence = confidence,
        )
        self._db.upsert_claim_node(node)
        return node

    def link_claims(
        self,
        src_id: str,
        dst_id: str,
        relation: ClaimRelation,
        weight: float = 1.0,
    ) -> ClaimEdge:
        edge = ClaimEdge(
            edge_id  = _new_id("ce"),
            src      = src_id,
            dst      = dst_id,
            relation = relation,
            weight   = weight,
        )
        self._db.upsert_claim_edge(edge)
        return edge

    def infer_relation(
        self, src: ClaimNode, dst: ClaimNode
    ) -> ClaimRelation:
        """Heuristic relation inference — replace with NLI model for precision."""
        sim = _cosine_sim(src.embedding, dst.embedding)
        src_l, dst_l = src.text.lower(), dst.text.lower()
        neg_words = ["not","no","never","false","wrong","contradict","refute","deny"]
        if any(w in dst_l for w in neg_words):
            return ClaimRelation.CONTRADICT
        if sim > 0.85:
            return ClaimRelation.SUPPORT
        if sim > 0.70:
            return ClaimRelation.REFINE
        return ClaimRelation.DERIVE

    def get_contradiction_chains(self, conv_id: str) -> List[Tuple[ClaimNode, ClaimNode]]:
        nodes = self._db.get_claim_nodes(conv_id)
        chains = []
        for node in nodes:
            edges = self._db.get_edges_for_node(node.node_id)
            for edge in edges:
                if edge.relation == ClaimRelation.CONTRADICT:
                    chains.append((node, node))   # simplified; real impl resolves both ends
        return chains

    def get_active_claims(self, conv_id: str, query: str, top_k: int = 8) -> List[ClaimNode]:
        nodes = self._db.get_claim_nodes(conv_id, limit=100)
        q_vec = self._emb.embed(query)
        scored = [(_cosine_sim(q_vec, n.embedding), n) for n in nodes]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]


class DebateMemoryManager:
    """
    Unified public API for all five memory layers.

    Typical orchestrator flow per turn
    -----------------------------------
    1.  manager.push_turn(conv_id, speaker, text, turn_idx)
    2.  ctx = manager.retrieve_context(conv_id, agent_id, current_text, budget)
    3.  agent generates response using ctx.to_prompt_block()
    4.  manager.process_turn(conv_id, agent_id, text, turn_idx)
        — extracts claims, detects boundaries, updates claim graph,
          persists episodes, deduplicates, compresses to semantic if needed
    5.  manager.update_belief(agent_id, ...)  [optional]
    6.  manager.add_global_fact(...)           [optional]
    """

    def __init__(
        self,
        db_path:            str = "./memory_store/debate_memory.db",
        embedder_model:     str = "all-MiniLM-L6-v2",
        user_id_prefix:     str = "debate",
        dedup_threshold:    float = DEDUP_THRESHOLD,
        compression_every:  int   = 6,   # compress episodic → semantic every N episodes
    ):
        self._prefix             = user_id_prefix
        self._compress_every     = compression_every

        self._embedder   = Embedder(embedder_model)
        self._db         = DatabaseBackend(db_path)
        self._dedup      = SemanticDeduplicator(dedup_threshold)
        self._retriever  = HybridRetriever(self._embedder)
        self._compressor = CompressionPipeline(self._embedder)
        self._graph      = ClaimGraphManager(self._db, self._embedder)

        self._dedup.rebuild_from_db(self._db)
        self._episode_counters: Dict[str, int] = {}

        logger.info(
            f"DebateMemoryManager ready | prefix={user_id_prefix} | "
            f"dedup_threshold={dedup_threshold}"
        )

    # L1 — Short-Term Working Memory

    def push_turn(
        self, conv_id: str, speaker: str, text: str, turn_index: int
    ):
        """Append a verbatim turn to the ring buffer."""
        self._db.push_turn(conv_id, speaker, text, turn_index)

    def get_recent_turns(
        self, conv_id: str, n: int = SHORT_TERM_WINDOW
    ) -> List[Dict]:
        return self._db.get_recent_turns(conv_id, n)

    # L2 — Episodic Memory

    def process_turn(
        self,
        conv_id:    str,
        agent_id:   str,
        text:       str,
        turn_index: int,
    ):
        """
        Full processing pipeline after a turn is generated:
          1. Extract claim nodes and link into graph
          2. Detect semantic boundary
          3. Persist episode event if boundary found
          4. Trigger semantic compression if threshold reached
        """
        # 1. Claim extraction
        claim_nodes = self._compressor.extract_claims(
            text, agent_id, turn_index, conv_id
        )
        inserted_ids: List[str] = []
        prev_node: Optional[ClaimNode] = None

        for node in claim_nodes:
            # Deduplicate
            if self._dedup.is_duplicate(MemoryLayer.EPISODIC.value, node.embedding):
                continue
            self._db.upsert_claim_node(node)
            self._dedup.register(MemoryLayer.EPISODIC.value, node.node_id, node.embedding)
            inserted_ids.append(node.node_id)

            # Auto-link sequential claims + detect contradictions vs active claims
            if prev_node:
                rel = self._graph.infer_relation(prev_node, node)
                self._graph.link_claims(prev_node.node_id, node.node_id, rel)
            else:
                # Link to top existing claim if available
                existing = self._db.get_claim_nodes(conv_id, limit=5)
                if existing:
                    rel = self._graph.infer_relation(existing[0], node)
                    self._graph.link_claims(existing[0].node_id, node.node_id, rel)

            prev_node = node

        # 2. Boundary detection
        boundary = self._compressor.detect_boundary(text)
        if boundary:
            recent = self._db.get_recent_turns(conv_id, n=5)
            evt = self._compressor.create_episode_event(
                event_type   = boundary,
                turns_window = recent,
                claim_ids    = inserted_ids,
                speaker      = agent_id,
                turn         = turn_index,
                conv_id      = conv_id,
            )
            # Deduplicate episode
            if not self._dedup.is_duplicate(MemoryLayer.EPISODIC.value, evt.embedding):
                self._db.upsert_episode(evt)
                self._dedup.register(MemoryLayer.EPISODIC.value, evt.event_id, evt.embedding)
                self._episode_counters[conv_id] = self._episode_counters.get(conv_id, 0) + 1

                # 3. Trigger semantic compression
                if self._episode_counters[conv_id] % self._compress_every == 0:
                    self._compress_episodes_to_semantic(conv_id)

    def _compress_episodes_to_semantic(self, conv_id: str):
        """Distil episodic events into a canonical semantic memory object."""
        episodes = self._db.get_all_episodes(conv_id)
        if len(episodes) < 3:
            return
        sem_mem = self._compressor.compress_to_semantic(episodes, conv_id)
        if not self._dedup.is_duplicate(MemoryLayer.SEMANTIC.value, sem_mem.embedding):
            self._db.upsert_memory(sem_mem)
            self._dedup.register(
                MemoryLayer.SEMANTIC.value, sem_mem.memory_id, sem_mem.embedding
            )
            logger.debug(f"Compressed {len(episodes)} episodes → {sem_mem.memory_id}")

    # L3 — Semantic Memory

    def add_semantic_fact(
        self,
        content:     str,
        speaker_id:  str  = "system",
        conv_id:     str  = "",
        confidence:  float = 1.0,
        metadata:    Optional[Dict] = None,
    ) -> Optional[MemoryObject]:
        emb = self._embedder.embed(content)
        if self._dedup.is_duplicate(MemoryLayer.SEMANTIC.value, emb):
            logger.debug("Semantic fact deduplicated (paraphrase exists)")
            return None
        obj = MemoryObject(
            memory_id  = _new_id("sem"),
            layer      = MemoryLayer.SEMANTIC,
            content    = content,
            embedding  = emb,
            provenance = Provenance(
                speaker_id      = speaker_id,
                turn_index      = -1,
                conversation_id = conv_id,
                confidence      = confidence,
            ),
            keywords   = _extract_keywords(content),
            metadata   = metadata or {},
        )
        self._db.upsert_memory(obj)
        self._dedup.register(MemoryLayer.SEMANTIC.value, obj.memory_id, emb)
        return obj

    def retrieve_semantic_facts(
        self, query: str, top_k: int = 5
    ) -> List[MemoryObject]:
        candidates = self._db.fetch_memories_by_layer(MemoryLayer.SEMANTIC, limit=300)
        return self._retriever.rank(query, candidates, top_k=top_k)

    # L4 — Agent-Private Memory

    def update_belief(
        self,
        agent_id:    str,
        content:     str,
        belief_type: str   = "ideological",
        confidence:  float = 1.0,
        conv_id:     str   = "",
        metadata:    Optional[Dict] = None,
    ) -> Optional[AgentBelief]:
        emb = self._embedder.embed(content)
        # Update existing if duplicate (revise, not duplicate)
        existing = self._db.get_beliefs(agent_id)
        for b in existing:
            if _cosine_sim(emb, b.embedding) >= DEDUP_THRESHOLD:
                b.confidence  = max(b.confidence, confidence)
                b.updated_at  = time.time()
                b.embedding   = emb
                self._db.upsert_belief(b)
                return b
        belief = AgentBelief(
            belief_id   = _new_id("bel"),
            agent_id    = agent_id,
            content     = content,
            belief_type = belief_type,
            embedding   = emb,
            confidence  = confidence,
            conv_id     = conv_id,
            metadata    = metadata or {},
        )
        self._db.upsert_belief(belief)
        return belief

    def get_agent_beliefs(
        self, agent_id: str, query: Optional[str] = None, top_k: int = 5
    ) -> List[AgentBelief]:
        all_beliefs = self._db.get_beliefs(agent_id)
        if not query or not all_beliefs:
            return all_beliefs[:top_k]
        q_vec = self._embedder.embed(query)
        scored = [(_cosine_sim(q_vec, b.embedding), b) for b in all_beliefs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:top_k]]

    def get_memory_statistics(self, agent_id: str) -> Dict[str, Any]:
        beliefs = self._db.get_beliefs(agent_id)
        sem     = self._db.fetch_memories_by_layer(MemoryLayer.SEMANTIC)
        return {
            "total_memories": len(beliefs) + len(sem),
            "beliefs":        len(beliefs),
            "semantic_facts": len(sem),
        }

    # L5 — Global Shared Memory

    def add_global_fact(
        self,
        content:    str,
        speaker_id: str   = "system",
        conv_id:    str   = "",
        confidence: float = 1.0,
        metadata:   Optional[Dict] = None,
    ) -> Optional[MemoryObject]:
        emb = self._embedder.embed(content)
        if self._dedup.is_duplicate(MemoryLayer.GLOBAL.value, emb):
            return None
        obj = MemoryObject(
            memory_id  = _new_id("glb"),
            layer      = MemoryLayer.GLOBAL,
            content    = content,
            embedding  = emb,
            provenance = Provenance(
                speaker_id      = speaker_id,
                turn_index      = -1,
                conversation_id = conv_id,
                confidence      = confidence,
            ),
            keywords   = _extract_keywords(content),
            metadata   = metadata or {},
        )
        self._db.upsert_memory(obj)
        self._dedup.register(MemoryLayer.GLOBAL.value, obj.memory_id, emb)
        return obj

    def retrieve_global_facts(
        self, query: str, top_k: int = 4
    ) -> List[MemoryObject]:
        candidates = self._db.fetch_memories_by_layer(MemoryLayer.GLOBAL, limit=200)
        return self._retriever.rank(query, candidates, top_k=top_k)

    # Context reconstruction (called by orchestrator before each turn)

    def retrieve_context(
        self,
        conv_id:     str,
        agent_id:    str,
        current_text: str,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> RetrievedContext:
        """
        Dynamically reconstruct the most relevant memory context for the
        next agent turn under a strict token budget.
        """
        budget_remaining = token_budget

        # --- L1: Short-term ring buffer ---
        short_turns = self._db.get_recent_turns(conv_id, n=SHORT_TERM_WINDOW)
        st_tokens = sum(_rough_tokens(t["text"]) for t in short_turns)
        budget_remaining -= st_tokens

        # --- L2: Active claims (graph + vector) ---
        raw_claims  = self._db.get_claim_nodes(conv_id, limit=60)
        top_claims  = self._retriever.rank_claims(current_text, raw_claims, top_k=6)
        budget_remaining -= sum(_rough_tokens(c.text) for c in top_claims)

        # --- L2: Unresolved episodes ---
        unresolved  = self._db.get_unresolved_episodes(conv_id, limit=20)
        top_eps     = self._retriever.rank_episodes(current_text, unresolved, top_k=4)
        budget_remaining -= sum(_rough_tokens(e.summary) for e in top_eps)

        # --- L4: Agent beliefs ---
        beliefs     = self.get_agent_beliefs(agent_id, query=current_text, top_k=4)
        budget_remaining -= sum(_rough_tokens(b.content) for b in beliefs)

        # --- L3: Semantic facts ---
        sem_k       = max(1, min(5, budget_remaining // 80))
        sem_facts   = self.retrieve_semantic_facts(current_text, top_k=sem_k)
        for m in sem_facts:
            self._db.increment_retrieval(m.memory_id)
        budget_remaining -= sum(_rough_tokens(m.content) for m in sem_facts)

        # --- L5: Global facts ---
        glb_k       = max(1, min(3, budget_remaining // 80))
        glb_facts   = self.retrieve_global_facts(current_text, top_k=glb_k)
        for m in glb_facts:
            self._db.increment_retrieval(m.memory_id)

        used_tokens = (token_budget - budget_remaining)

        return RetrievedContext(
            short_term_turns    = short_turns,
            active_claims       = top_claims,
            unresolved_episodes = top_eps,
            agent_beliefs       = beliefs,
            semantic_facts      = sem_facts,
            global_facts        = glb_facts,
            token_estimate      = used_tokens,
        )


    def get_debate_analytics(self, conv_id: str) -> Dict[str, Any]:
        """Long-horizon conversational analytics."""
        episodes  = self._db.get_all_episodes(conv_id)
        claims    = self._db.get_claim_nodes(conv_id, limit=500)
        unresolved = [e for e in episodes if not e.resolved]

        # Speaker distribution
        speaker_counts: Dict[str, int] = {}
        for c in claims:
            speaker_counts[c.speaker] = speaker_counts.get(c.speaker, 0) + 1

        # Event type distribution
        event_dist: Dict[str, int] = {}
        for e in episodes:
            event_dist[e.event_type.value] = event_dist.get(e.event_type.value, 0) + 1

        return {
            "conversation_id":        conv_id,
            "total_claims":           len(claims),
            "total_episodes":         len(episodes),
            "unresolved_count":       len(unresolved),
            "speaker_distribution":   speaker_counts,
            "event_type_distribution": event_dist,
            "semantic_memory_count":  len(self._db.fetch_memories_by_layer(MemoryLayer.SEMANTIC)),
            "global_memory_count":    len(self._db.fetch_memories_by_layer(MemoryLayer.GLOBAL)),
        }

    # Episode reconstruction for long-horizon reasoning
    def reconstruct_episode_narrative(self, conv_id: str) -> str:
        """
        Build a coherent narrative summary of the entire conversation's
        episodic trajectory. Useful for resuming long-running debates.
        """
        episodes = self._db.get_all_episodes(conv_id)
        if not episodes:
            return "(no episodic history)"
        parts = []
        for ep in sorted(episodes, key=lambda e: e.provenance.timestamp):
            ts = datetime.fromtimestamp(
                ep.provenance.timestamp, tz=timezone.utc
            ).strftime("%H:%M:%S")
            marker = "done" if ep.resolved else "○"
            parts.append(f"  {marker} [{ts}] {ep.event_type.value}: {ep.summary[:120]}")
        return "\n".join(parts)
