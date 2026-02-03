"""
TTS (Text-to-Speech) Module

Integrates Kokoro TTS for real-time audio generation during debates.
Generates audio for each agent utterance and combines them into a single file.
"""

import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, List, Literal
from kokoro import KPipeline
import torch

logger = logging.getLogger(__name__)


class TTSConfig:
    """Configuration for TTS generation."""
    
    def __init__(
        self,
        agent_1_voice: str = "af_bella",
        agent_2_voice: str = "am_adam",
        sample_rate: int = 24000,
        enable_realtime: bool = True
    ):
        """
        Initialize TTS configuration.
        
        Args:
            agent_1_voice: Voice for agent 1
            agent_2_voice: Voice for agent 2
            sample_rate: Audio sample rate
            enable_realtime: Whether to generate audio in real-time
        """
        self.agent_1_voice = agent_1_voice
        self.agent_2_voice = agent_2_voice
        self.sample_rate = sample_rate
        self.enable_realtime = enable_realtime


class DebateTTSGenerator:
    """
    Generates TTS audio for debate conversations using Kokoro.
    """
    
    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        output_dir: str = "./audio_outputs"
    ):
        """
        Initialize the TTS generator.
        
        Args:
            config: TTS configuration
            output_dir: Directory for audio outputs
        """
        self.config = config or TTSConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kokoro pipeline
        try:
            logger.info("Initializing Kokoro TTS pipeline...")
            self.pipeline = KPipeline(lang_code='a')
            logger.info("✓ Kokoro TTS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro: {e}", exc_info=True)
            raise
        
        # Storage for audio segments
        self.audio_segments: List[np.ndarray] = []
        self.segment_metadata: List[dict] = []
    
    def get_voice_for_agent(
        self,
        agent_id: Literal["agent_1", "agent_2"]
    ) -> str:
        """
        Get the voice configuration for a specific agent.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Voice name for the agent
        """
        if agent_id == "agent_1":
            return self.config.agent_1_voice
        else:
            return self.config.agent_2_voice
    
    def generate_utterance_audio(
        self,
        text: str,
        agent_id: Literal["agent_1", "agent_2"],
        turn_number: int
    ) -> np.ndarray:
        """
        Generate audio for a single utterance.
        
        Args:
            text: Text to convert to speech
            agent_id: Which agent is speaking
            turn_number: Turn number in the conversation
        
        Returns:
            Audio data as numpy array
        """
        voice = self.get_voice_for_agent(agent_id)
        
        logger.info(
            f"Generating TTS for {agent_id} (turn {turn_number}) "
            f"with voice '{voice}'"
        )
        
        try:
            # Generate audio using Kokoro
            generator = self.pipeline(text, voice=voice)
            
            # Collect all audio segments from the generator
            audio_parts = []
            for i, (gs, ps, audio) in enumerate(generator):
                logger.debug(
                    f"Generated segment {i} for turn {turn_number}: "
                    f"gs={gs}, ps={ps}"
                )
                audio_parts.append(audio)
            
            # Concatenate all parts
            if audio_parts:
                full_audio = np.concatenate(audio_parts)
                
                # Store metadata
                self.segment_metadata.append({
                    "turn_number": turn_number,
                    "agent_id": agent_id,
                    "voice": voice,
                    "text_length": len(text),
                    "audio_length": len(full_audio),
                    "duration_seconds": len(full_audio) / self.config.sample_rate
                })
                
                logger.info(
                    f"✓ Generated {len(full_audio)} samples "
                    f"({len(full_audio) / self.config.sample_rate:.2f}s) "
                    f"for turn {turn_number}"
                )
                
                return full_audio
            else:
                logger.warning(f"No audio generated for turn {turn_number}")
                return np.array([], dtype=np.float32)
                
        except Exception as e:
            logger.error(
                f"Error generating TTS for turn {turn_number}: {e}",
                exc_info=True
            )
            # Return silence on error
            return np.array([], dtype=np.float32)
    
    def add_silence(self, duration_seconds: float = 0.5) -> np.ndarray:
        """
        Generate silence between utterances.
        
        Args:
            duration_seconds: Duration of silence
        
        Returns:
            Silence audio array
        """
        num_samples = int(duration_seconds * self.config.sample_rate)
        return np.zeros(num_samples, dtype=np.float32)
    
    def add_utterance(
        self,
        text: str,
        agent_id: Literal["agent_1", "agent_2"],
        turn_number: int,
        add_pause: bool = True
    ) -> None:
        """
        Generate and store audio for an utterance.
        
        Args:
            text: Utterance text
            agent_id: Speaking agent
            turn_number: Turn number
            add_pause: Whether to add silence after utterance
        """
        # Generate audio
        audio = self.generate_utterance_audio(text, agent_id, turn_number)
        
        if len(audio) > 0:
            # Add to segments
            self.audio_segments.append(audio)
            
            # Add pause between utterances
            if add_pause:
                silence = self.add_silence(0.5)
                self.audio_segments.append(silence)
    
    def combine_all_audio(self) -> np.ndarray:
        """
        Combine all audio segments into one array.
        
        Returns:
            Combined audio data
        """
        if not self.audio_segments:
            logger.warning("No audio segments to combine")
            return np.array([], dtype=np.float32)
        
        logger.info(f"Combining {len(self.audio_segments)} audio segments...")
        combined = np.concatenate(self.audio_segments)
        
        total_duration = len(combined) / self.config.sample_rate
        logger.info(
            f"✓ Combined audio: {len(combined)} samples, "
            f"{total_duration:.2f} seconds"
        )
        
        return combined
    
    def save_debate_audio(
        self,
        filename: Optional[str] = None,
        topic: Optional[str] = None
    ) -> str:
        """
        Save the complete debate audio to a WAV file.
        
        Args:
            filename: Optional custom filename
            topic: Debate topic (used for auto-naming)
        
        Returns:
            Path to saved audio file
        """
        # Generate filename if not provided
        if filename is None:
            if topic:
                safe_topic = "".join(
                    c if c.isalnum() or c in (' ', '-', '_') else '_'
                    for c in topic
                )[:50]
                filename = f"debate_{safe_topic}.wav"
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"debate_{timestamp}.wav"
        
        # Ensure .wav extension
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        filepath = self.output_dir / filename
        
        # Combine all audio
        combined_audio = self.combine_all_audio()
        
        if len(combined_audio) == 0:
            logger.warning("No audio to save")
            return ""
        
        # Save to file
        logger.info(f"Saving debate audio to: {filepath}")
        sf.write(
            str(filepath),
            combined_audio,
            self.config.sample_rate
        )
        
        # Log summary
        total_duration = len(combined_audio) / self.config.sample_rate
        logger.info(
            f"✓ Saved debate audio: {filepath}\n"
            f"  Duration: {total_duration:.2f} seconds\n"
            f"  Turns: {len(self.segment_metadata)}\n"
            f"  Sample rate: {self.config.sample_rate} Hz"
        )
        
        return str(filepath)
    
    def save_individual_utterances(
        self,
        prefix: str = "turn"
    ) -> List[str]:
        """
        Save each utterance as a separate WAV file.
        
        Args:
            prefix: Filename prefix
        
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        segment_idx = 0
        for i, audio in enumerate(self.audio_segments):
            # Skip silence segments
            if np.all(audio == 0):
                continue
            
            if segment_idx < len(self.segment_metadata):
                metadata = self.segment_metadata[segment_idx]
                turn_num = metadata['turn_number']
                agent_id = metadata['agent_id']
                
                filename = f"{prefix}_{turn_num:03d}_{agent_id}.wav"
                filepath = self.output_dir / filename
                
                sf.write(str(filepath), audio, self.config.sample_rate)
                saved_files.append(str(filepath))
                
                logger.debug(f"Saved utterance: {filepath}")
                segment_idx += 1
        
        logger.info(f"✓ Saved {len(saved_files)} individual utterances")
        return saved_files
    
    def get_statistics(self) -> dict:
        """
        Get statistics about generated audio.
        
        Returns:
            Dictionary of statistics
        """
        combined = self.combine_all_audio()
        total_duration = len(combined) / self.config.sample_rate
        
        agent_1_turns = sum(
            1 for m in self.segment_metadata if m['agent_id'] == 'agent_1'
        )
        agent_2_turns = sum(
            1 for m in self.segment_metadata if m['agent_id'] == 'agent_2'
        )
        
        agent_1_duration = sum(
            m['duration_seconds'] 
            for m in self.segment_metadata 
            if m['agent_id'] == 'agent_1'
        )
        agent_2_duration = sum(
            m['duration_seconds'] 
            for m in self.segment_metadata 
            if m['agent_id'] == 'agent_2'
        )
        
        return {
            "total_duration": total_duration,
            "total_turns": len(self.segment_metadata),
            "agent_1_turns": agent_1_turns,
            "agent_2_turns": agent_2_turns,
            "agent_1_duration": agent_1_duration,
            "agent_2_duration": agent_2_duration,
            "total_samples": len(combined),
            "sample_rate": self.config.sample_rate
        }
    
    def clear(self) -> None:
        """Clear all stored audio segments."""
        self.audio_segments.clear()
        self.segment_metadata.clear()
        logger.info("Cleared all audio segments")
