"""
Utilities Module

Provides logging configuration, persistence utilities, and helper functions
for the multi-agent debate system.
"""

import logging
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from conversation_state import ConversationState


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Configure logging for the debate system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        log_format: Optional custom log format
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    format_string = log_format or (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    # Set specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}")


class ConversationPersistence:
    """
    Handles persistence of conversation states to disk.
    """
    
    def __init__(self, storage_dir: str = "./conversations"):
        """
        Initialize persistence handler.
        
        Args:
            storage_dir: Directory for storing conversation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized persistence at: {self.storage_dir}")
    
    def _generate_filename(self, topic: str, timestamp: datetime) -> str:
        """
        Generate a filename for a conversation.
        
        Args:
            topic: Debate topic
            timestamp: Conversation start time
        
        Returns:
            Filename string
        """
        # Sanitize topic for filename
        safe_topic = "".join(
            c if c.isalnum() or c in (' ', '-', '_') else '_'
            for c in topic
        )[:50]
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"debate_{timestamp_str}_{safe_topic}.json"
    
    def save_conversation(
        self,
        conversation_state: ConversationState,
        filename: Optional[str] = None
    ) -> str:
        """
        Save a conversation state to disk.
        
        Args:
            conversation_state: ConversationState to save
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = self._generate_filename(
                conversation_state.topic,
                conversation_state.conversation_start_time
            )
        
        filepath = self.storage_dir / filename
        
        # Convert to dictionary
        data = conversation_state.to_dict()
        
        # Add metadata
        data["_metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved conversation to: {filepath}")
        return str(filepath)
    
    def save_transcript(
        self,
        conversation_state: ConversationState,
        filename: Optional[str] = None
    ) -> str:
        """
        Save just the transcript as a readable text file.
        
        Args:
            conversation_state: ConversationState to save
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            base_filename = self._generate_filename(
                conversation_state.topic,
                conversation_state.conversation_start_time
            )
            filename = base_filename.replace('.json', '.txt')
        
        filepath = self.storage_dir / filename
        
        # Get transcript text
        transcript_text = conversation_state.get_transcript_text()
        
        # Add metadata header
        metrics = conversation_state.compute_metrics()
        header = [
            "=" * 80,
            "DEBATE TRANSCRIPT",
            "=" * 80,
            f"Topic: {conversation_state.topic}",
            f"Start Time: {conversation_state.conversation_start_time.isoformat()}",
            f"Total Turns: {metrics.total_turns}",
            f"Duration: {metrics.conversation_duration_seconds:.1f} seconds",
            f"Status: {'Terminated' if conversation_state.is_terminated() else 'Active'}",
            "=" * 80,
            ""
        ]
        
        full_text = "\n".join(header) + "\n" + transcript_text
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        self.logger.info(f"Saved transcript to: {filepath}")
        return str(filepath)
    
    def load_conversation(self, filename: str) -> Dict[str, Any]:
        """
        Load a saved conversation from disk.
        
        Args:
            filename: Filename or path to conversation file
        
        Returns:
            Conversation data as dictionary
        """
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Conversation file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded conversation from: {filepath}")
        return data
    
    def list_conversations(self) -> list[str]:
        """
        List all saved conversations.
        
        Returns:
            List of conversation filenames
        """
        json_files = sorted(self.storage_dir.glob("*.json"))
        return [f.name for f in json_files]


class MetricsTracker:
    """
    Tracks and computes various metrics during conversation execution.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.logger = logging.getLogger(__name__)
        self.event_log = []
    
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Log an event for metrics tracking.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.event_log.append(event)
        self.logger.debug(f"Logged event: {event_type}")
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics from logged events.
        
        Returns:
            Dictionary of computed statistics
        """
        if not self.event_log:
            return {}
        
        stats = {
            "total_events": len(self.event_log),
            "event_types": {},
            "timeline": []
        }
        
        # Count event types
        for event in self.event_log:
            event_type = event["event_type"]
            stats["event_types"][event_type] = (
                stats["event_types"].get(event_type, 0) + 1
            )
        
        # Build timeline
        for event in self.event_log:
            stats["timeline"].append({
                "timestamp": event["timestamp"],
                "event_type": event["event_type"],
                "summary": str(event["data"])[:100]
            })
        
        return stats
    
    def export_events(self, filepath: str) -> None:
        """
        Export event log to JSON file.
        
        Args:
            filepath: Path to output file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.event_log, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(self.event_log)} events to: {filepath}")


def validate_api_key(api_key: str) -> bool:
    """
    Validate Groq API key format.
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if format is valid
    """
    if not api_key:
        return False
    
    # Basic validation - Groq keys typically start with 'gsk_'
    return isinstance(api_key, str) and len(api_key) > 20


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def get_environment_variable(
    var_name: str,
    default: Optional[str] = None,
    required: bool = False
) -> Optional[str]:
    """
    Get environment variable with optional default and requirement.
    
    Args:
        var_name: Environment variable name
        default: Default value if not found
        required: Whether variable is required
    
    Returns:
        Variable value or default
    
    Raises:
        ValueError: If required variable is not found
    """
    value = os.environ.get(var_name, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable not set: {var_name}")
    
    return value


class DebateFormatter:
    """
    Formats debate output for various display purposes.
    """
    
    @staticmethod
    def format_for_console(conversation_state: ConversationState) -> str:
        """
        Format conversation for console display with colors.
        
        Args:
            conversation_state: Conversation to format
        
        Returns:
            Formatted string for console
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"DEBATE: {conversation_state.topic}")
        lines.append("=" * 80)
        lines.append("")
        
        # Transcript
        for turn in conversation_state.get_full_transcript():
            agent_label = turn.agent_id.replace("_", " ").title()
            lines.append(f"{agent_label} (Turn {turn.turn_number}):")
            lines.append(f"  {turn.utterance}")
            lines.append("")
        
        # Footer with metrics
        metrics = conversation_state.compute_metrics()
        lines.append("=" * 80)
        lines.append(f"Total Turns: {metrics.total_turns}")
        lines.append(
            f"Duration: {format_duration(metrics.conversation_duration_seconds)}"
        )
        lines.append(f"Status: {'Terminated' if conversation_state.is_terminated() else 'Active'}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_for_html(conversation_state: ConversationState) -> str:
        """
        Format conversation as HTML.
        
        Args:
            conversation_state: Conversation to format
        
        Returns:
            HTML string
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            f"<title>Debate: {conversation_state.topic}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".header { background: #333; color: white; padding: 20px; }",
            ".turn { margin: 20px 0; padding: 15px; border-left: 3px solid #007bff; }",
            ".agent1 { border-color: #007bff; }",
            ".agent2 { border-color: #28a745; }",
            ".metrics { background: #f8f9fa; padding: 15px; margin-top: 20px; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='header'>",
            f"<h1>Debate: {conversation_state.topic}</h1>",
            "</div>"
        ]
        
        # Transcript
        for turn in conversation_state.get_full_transcript():
            agent_class = turn.agent_id.replace("_", "")
            agent_label = turn.agent_id.replace("_", " ").title()
            html_parts.append(f"<div class='turn {agent_class}'>")
            html_parts.append(f"<strong>{agent_label} (Turn {turn.turn_number}):</strong>")
            html_parts.append(f"<p>{turn.utterance}</p>")
            html_parts.append("</div>")
        
        # Metrics
        metrics = conversation_state.compute_metrics()
        html_parts.append("<div class='metrics'>")
        html_parts.append(f"<p><strong>Total Turns:</strong> {metrics.total_turns}</p>")
        html_parts.append(
            f"<p><strong>Duration:</strong> "
            f"{format_duration(metrics.conversation_duration_seconds)}</p>"
        )
        html_parts.append(
            f"<p><strong>Status:</strong> "
            f"{'Terminated' if conversation_state.is_terminated() else 'Active'}</p>"
        )
        html_parts.append("</div>")
        
        html_parts.append("</body>")
        html_parts.append("</html>")
        
        return "\n".join(html_parts)
