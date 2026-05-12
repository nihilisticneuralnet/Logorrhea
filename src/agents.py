from typing import Dict, Literal, Optional
from abc import ABC, abstractmethod
import logging
from groq import Groq
import torch

logger = logging.getLogger(__name__)

PROVIDER_GROQ = "groq"
PROVIDER_HF   = "huggingface"

class HFLocalBackend:
    _instances: dict = {}  

    def __new__(cls, model_id: str, **kwargs):
        if model_id in cls._instances:
            logger.info(f"[HFLocalBackend] Reusing loaded model: {model_id}")
            return cls._instances[model_id]
        instance = super().__new__(cls)
        cls._instances[model_id] = instance
        return instance

    def __init__(
        self,
        model_id:         str,
        hf_token:         Optional[str] = None,
        load_in_4bit:     bool          = False,   # QLoRA-style 4-bit (needs bitsandbytes)
        load_in_8bit:     bool          = False,   # LLM.int8() (needs bitsandbytes)
        compile_model:    bool          = True,    # torch.compile forward pass
        max_new_tokens:   int           = 1024,    # hard cap stored on backend
    ):
        if hasattr(self, "_ready"):          # already initialised by __new__ cache hit
            return

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
                TextIteratorStreamer,
            )
        except ImportError:
            raise ImportError(
                "transformers>=4.40 is required for local HF inference. "
                "Install: pip install transformers accelerate bitsandbytes"
            )

        self.model_id       = model_id
        self.max_new_tokens = max_new_tokens

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit                = True,
                bnb_4bit_compute_dtype      = dtype,
                bnb_4bit_use_double_quant   = True,   # nested quantization
                bnb_4bit_quant_type         = "nf4",  # NormalFloat4 — best quality
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        attn_impl = "eager"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            logger.info("[HFLocalBackend] Flash Attention 2 enabled")
        except ImportError:
            logger.info(
                "[HFLocalBackend] flash_attn not found — "
                "falling back to eager attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

        logger.info(f"[HFLocalBackend] Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token           = hf_token,
            use_fast        = True,      # Rust-based fast tokenizer
            padding_side    = "left",    # required for batch decode; harmless for single
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            f"[HFLocalBackend] Loading model: {model_id} "
            f"| dtype={dtype} | attn={attn_impl} "
            f"| 4bit={load_in_4bit} | 8bit={load_in_8bit}"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token                  = hf_token,
            torch_dtype            = dtype if not bnb_config else None,
            quantization_config    = bnb_config,
            device_map             = "auto",     # handles multi-GPU / CPU offload
            attn_implementation    = attn_impl,
            low_cpu_mem_usage      = True,       # stream weights during load
        )
        self.model.eval()

        if compile_model and not (load_in_4bit or load_in_8bit):
            logger.info("[HFLocalBackend] Compiling model with torch.compile …")
            # "reduce-overhead" fuses small ops; "max-autotune" is slower to compile
            # but faster at runtime — use max-autotune for long debates
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("[HFLocalBackend] torch.compile done")

        self.device = next(
            (p.device for p in self.model.parameters()), torch.device("cpu")
        )

        self._StreamerClass = TextIteratorStreamer

        self._ready = True
        logger.info(f"[HFLocalBackend] Model ready on {self.device}")

    def generate(
        self,
        messages:    list,
        temperature: float,
        max_tokens:  int,
        stream:      bool = True,
    ) -> str:
        import threading

        # Most modern checkpoints (Llama-3, Mistral, Qwen …) ship a
        # tokenizer.chat_template that handles system/user/assistant roles.
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize         = True,
            add_generation_prompt = True,
            return_tensors   = "pt",
        ).to(self.device)

        gen_kwargs = dict(
            input_ids      = input_ids,
            max_new_tokens = min(max_tokens, self.max_new_tokens),
            temperature    = max(temperature, 1e-2),  # avoid 0 → greedy issues
            do_sample      = temperature > 1e-2,
            repetition_penalty = 1.1,   # mild penalty to reduce loops
            pad_token_id   = self.tokenizer.pad_token_id,
            eos_token_id   = self.tokenizer.eos_token_id,
            use_cache      = True,      # keep KV-cache alive during generation
        )

        if stream:
            # Streamer path (lower first-token latency)
            streamer = self._StreamerClass(
                self.tokenizer,
                skip_prompt            = True,   # don't echo the input back
                skip_special_tokens    = True,
            )
            gen_kwargs["streamer"] = streamer

            # Run generation in a background thread so we can iterate the
            # streamer in the main thread without blocking
            thread = threading.Thread(
                target = self.model.generate,
                kwargs = gen_kwargs,
                daemon = True,
            )
            thread.start()

            parts = []
            for token_text in streamer:
                parts.append(token_text)
            thread.join()

            return "".join(parts).strip()

        else:
            # Non-streaming path (simpler, slightly higher latency)
            with torch.inference_mode():
                output_ids = self.model.generate(**gen_kwargs)

            # Decode only the newly generated tokens
            new_ids = output_ids[0, input_ids.shape[-1]:]
            return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

class Agent(ABC):
    def __init__(
        self,
        agent_id:    Literal["agent_1", "agent_2"],
        api_key:     str,
        stance:      Optional[str] = None,
        personality: Optional[str] = None,
        provider:    str           = PROVIDER_GROQ,
        hf_model_id: Optional[str] = None,   # e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    ):
        self.agent_id    = agent_id
        self.stance      = stance
        self.personality = personality
        self.provider    = provider

        if provider == PROVIDER_GROQ:
            self.client = Groq(api_key=api_key)

        elif provider == PROVIDER_HF:
            # Lazy-import so Groq-only installs are unaffected
            try:
                from huggingface_hub import InferenceClient
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for the HuggingFace provider. "
                    "Install it with: pip install huggingface-hub"
                )
            model = hf_model_id or "meta-llama/Meta-Llama-3-8B-Instruct"
            self.client   = InferenceClient(model=model, token=api_key)
            self.hf_model = model

        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose '{PROVIDER_GROQ}' or '{PROVIDER_HF}'."
            )

        logger.info(
            f"Initialized {agent_id} | provider={provider} | "
            f"stance={stance} | personality={personality}"
        )

    @abstractmethod
    def generate_utterance(
        self,
        context:    Dict,
        temperature: float,
        model:       str,
        max_tokens:  int,
    ) -> str:
        pass


class DebateAgent(Agent):
    def __init__(
        self,
        agent_id:               Literal["agent_1", "agent_2"],
        api_key:                str,
        stance:                 Optional[str] = None,
        personality:            Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        provider:               str           = PROVIDER_GROQ,
        hf_model_id:            Optional[str] = None,
    ):
        super().__init__(
            agent_id    = agent_id,
            api_key     = api_key,
            stance      = stance,
            personality = personality,
            provider    = provider,
            hf_model_id = hf_model_id,
        )
        self.system_prompt_template = (
            system_prompt_template or self._default_system_prompt()
        )
        if provider == PROVIDER_HF:
            self._hf_backend = HFLocalBackend(
                model_id      = hf_model_id or "meta-llama/Meta-Llama-3-8B-Instruct",
                hf_token      = api_key,
                load_in_4bit  = hf_load_in_4bit,    # pass through from __init__ args
                load_in_8bit  = hf_load_in_8bit,
                compile_model = hf_compile,
            )
    
    def _default_system_prompt(self) -> str:
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
        history_lines = [f"DEBATE TOPIC: {context['topic']}\n"]
        
        for turn in context['transcript']:
            agent_label = turn['agent_id'].replace('_', ' ').title()
            history_lines.append(f"{agent_label}: {turn['utterance']}")
        
        return "\n\n".join(history_lines)
    
    def _construct_prompt(self, context: Dict) -> str:
        if not context['transcript']:
            prompt = (
                f"You are beginning a debate on the following topic:\n\n"
                f"{context['topic']}\n\n"
                f"Provide your opening statement. Present your position clearly "
                f"and set the stage for a thoughtful discussion."
            )
        else:
            history = self._build_conversation_history(context)
            prompt = (
                f"{history}\n\n"
                f"It is now your turn. Respond to the previous argument and "
                f"advance your position. Be direct, logical, and engaging."
            )
        
        return prompt
    
    def generate_utterance(
        self,
        context:     Dict,
        temperature: float,
        model:       str,
        max_tokens:  int,
    ) -> str:
        logger.info(
            f"{self.agent_id} generating utterance "
            f"(turn {context['turn_number'] + 1}) via {self.provider}"
        )

        if not context["is_my_turn"]:
            raise ValueError(
                f"{self.agent_id} attempted to speak but it's not their turn"
            )

        user_prompt = self._construct_prompt(context)

        try:
            if self.provider == PROVIDER_GROQ:
                return self._generate_groq(user_prompt, temperature, model, max_tokens)
            else:
                return self._generate_hf(user_prompt, temperature, max_tokens)

        except Exception as e:
            logger.error(
                f"Error generating utterance for {self.agent_id}: {e}",
                exc_info=True,
            )
            return (
                "I apologize, I'm experiencing technical difficulties "
                "and cannot respond properly at this moment."
            )

    def _generate_groq(
        self,
        user_prompt: str,
        temperature: float,
        model:       str,
        max_tokens:  int,
    ) -> str:
        completion = self.client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user",   "content": user_prompt},
            ],
            temperature           = temperature,
            max_completion_tokens = max_tokens,
            top_p                 = 1,
            stream                = True,
            stop                  = None,
        )

        parts = [
            chunk.choices[0].delta.content
            for chunk in completion
            if chunk.choices[0].delta.content
        ]
        return "".join(parts).strip() or "I need more time to formulate my response."

    def _generate_hf(
        self,
        user_prompt: str,
        temperature: float,
        max_tokens:  int,
    ) -> str:
        """
        Generate via the singleton HFLocalBackend.
        The backend is loaded once; subsequent calls reuse the warm model.
        """
        utterance = self._hf_backend.generate(
            messages = [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = temperature,
            max_tokens  = max_tokens,
            stream      = True,
        )
        return utterance or "I need more time to formulate my response."

class AgentFactory:
    @staticmethod
    def create_debate_pair(
        api_key:              str,
        topic:                str,
        agent_1_stance:       Optional[str] = None,
        agent_2_stance:       Optional[str] = None,
        agent_1_personality:  Optional[str] = None,
        agent_2_personality:  Optional[str] = None,
        provider:             str           = PROVIDER_GROQ,   
        hf_model_id:          Optional[str] = None,           
    ) -> tuple[DebateAgent, DebateAgent]:
        shared = dict(provider=provider, hf_model_id=hf_model_id)

        agent_1 = DebateAgent(
            agent_id    = "agent_1",
            api_key     = api_key,
            stance      = agent_1_stance      or "Support the affirmative position",
            personality = agent_1_personality or (
                "Analytical and evidence-focused, favouring logical arguments "
                "and empirical data"
            ),
            **shared,
        )
        agent_2 = DebateAgent(
            agent_id    = "agent_2",
            api_key     = api_key,
            stance      = agent_2_stance      or "Support the negative position",
            personality = agent_2_personality or (
                "Philosophical and principle-based, emphasising ethical "
                "considerations and thought experiments"
            ),
            **shared,
        )

        logger.info(f"Created agent pair | provider={provider} | topic='{topic}'")
        return agent_1, agent_2
    
    @staticmethod
    def create_specialized_agents(
        api_key: str,
        agent_1_config: Dict,
        agent_2_config: Dict
    ) -> tuple[DebateAgent, DebateAgent]:
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
