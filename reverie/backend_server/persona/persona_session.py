"""
PersonaSession - Manages persistent Claude CLI sessions per persona

This is a fundamental architectural departure from the original Generative Agents:
- Original: Stateless API calls, context reconstructed each time
- Claudeville: Each persona IS a persistent Claude session with genuine continuity

The persona maintains identity and memory through the Claude session itself,
rather than reconstructing context from embeddings on each call. This enables
more authentic agent behavior with true continuity of experience.

Author: Claudeville Project
"""

import json
import math
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any, Optional


class PersonaSession:
    """
    Manages a persistent Claude CLI session for a single persona.

    This class is the bridge between the simulation and Claude's persistent
    session capabilities. Each persona maintains their own session, which
    preserves context and identity across interactions.

    Key features:
    - Persistent session management via Claude CLI
    - Automatic context compaction at 75% capacity
    - Memory injection and retrieval
    - Token usage tracking

    Attributes:
        persona: The Persona instance this session manages
        session_id: Unique identifier for this Claude session
        is_new_session: Whether this is a fresh session needing priming
        token_usage: Current estimated token usage in the session
        message_count: Number of messages sent in this session
        subconscious: Reference to SubconsciousRetriever for memory access
        recent_interactions: Set of persona names interacted with this session
        visited_locations: Set of locations visited this session
    """

    # Context limits for Claude Opus
    MAX_CONTEXT_TOKENS = 200000  # Opus context window
    COMPACTION_THRESHOLD = 0.75  # Trigger compaction at 75% (150K tokens)

    # CLI command timeout in seconds
    CLI_TIMEOUT = 120

    def __init__(self, persona):
        """
        Initialize a PersonaSession for the given persona.

        Args:
            persona: The Persona instance to manage a session for.
                    Must have 'name' and 'scratch' attributes.
        """
        self.persona = persona
        self.session_id = self._load_or_create_session_id()
        self.is_new_session = False  # Will be set True if new session created
        self.token_usage = 0
        self.message_count = 0
        self.subconscious = None  # Set by persona after initialization
        self.recent_interactions: set[str] = set()  # Persons seen this session
        self.visited_locations: set[str] = set()  # Locations visited this session

    def _get_storage_base_path(self) -> Optional[str]:
        """
        Get the base storage path for this persona.

        The storage structure follows:
        environment/frontend_server/storage/{simulation}/personas/{name}/

        Returns:
            The base storage path for the persona, or None if not determinable.
        """
        # Try to get the persona's folder_mem path if available
        # The persona is initialized with folder_mem_saved which contains the path
        if hasattr(self.persona, "a_mem") and hasattr(self.persona.a_mem, "embeddings"):
            # We can try to infer from the associative memory path
            # but that's stored internally. Let's check if scratch has storage info
            pass

        # Try to construct path from persona name and check common locations
        # In the original codebase, personas are stored under:
        # environment/frontend_server/storage/{sim_name}/personas/{persona_name}/

        # Get the directory of this file to navigate to storage
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to reverie/backend_server, then to environment/frontend_server/storage
        base_path = os.path.join(
            current_dir, "..", "..", "..", "environment", "frontend_server", "storage"
        )
        base_path = os.path.normpath(base_path)

        if not os.path.exists(base_path):
            return None

        # Look for the persona in any simulation folder
        persona_name = self.persona.name if hasattr(self.persona, "name") else None
        if not persona_name:
            persona_name = (
                self.persona.scratch.name if hasattr(self.persona, "scratch") else None
            )

        if not persona_name:
            return None

        # Search through simulation folders to find this persona
        for sim_folder in os.listdir(base_path):
            sim_path = os.path.join(base_path, sim_folder)
            if os.path.isdir(sim_path):
                personas_path = os.path.join(sim_path, "personas", persona_name)
                if os.path.exists(personas_path):
                    return personas_path

        return None

    def _get_session_state_path(self) -> Optional[str]:
        """
        Get path to session_state.json for this persona.

        The session state file stores:
        - session_id: Current Claude session ID
        - estimated_tokens: Estimated token usage
        - last_compaction: Timestamp of last compaction
        - compaction_count: Number of times compacted
        - compaction_summaries: List of compaction summaries
        - recent_interactions: List of recently interacted personas
        - visited_locations: List of visited locations

        Returns:
            Path to session_state.json, or None if storage path not found.
        """
        storage_path = self._get_storage_base_path()
        if storage_path:
            return os.path.join(storage_path, "session_state.json")
        return None

    def _load_or_create_session_id(self) -> str:
        """
        Load existing session ID from state file or create new one.

        If a session_state.json exists and contains a valid session_id,
        that ID is returned. Otherwise, a new UUID is generated and
        is_new_session is set to True.

        Returns:
            The session ID (either existing or newly created).
        """
        state_path = self._get_session_state_path()
        if state_path and os.path.exists(state_path):
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                    existing_id = state.get("session_id")
                    if existing_id:
                        # Load other state
                        self.token_usage = state.get("estimated_tokens", 0)
                        self.recent_interactions = set(
                            state.get("recent_interactions", [])
                        )
                        self.visited_locations = set(state.get("visited_locations", []))
                        return existing_id
            except (OSError, json.JSONDecodeError):
                pass

        # No valid existing session, create new one
        self.is_new_session = True
        return str(uuid.uuid4())

    def _build_system_prompt(self) -> str:
        """
        Build character identity prompt from persona data.

        This creates the system prompt that establishes the persona's identity
        in the Claude session. It includes their name, core traits, current
        situation, and temporal/spatial context.

        Returns:
            A formatted system prompt string for Claude.
        """
        scratch = self.persona.scratch

        # Get the identity stable set (ISS) which contains core persona info
        iss = scratch.get_str_iss() if hasattr(scratch, "get_str_iss") else ""

        # Format current time safely
        curr_time_str = "unknown"
        if scratch.curr_time:
            curr_time_str = scratch.curr_time.strftime("%A %B %d, %Y - %H:%M")

        return f"""You are {scratch.name}.

{iss}

You exist in a simulated town called Smallville. You have your own
thoughts, memories, and agency. Respond naturally as yourself.

Current situation: {scratch.act_description or "idle"}
Current time: {curr_time_str}
Current location: {scratch.act_address or "unknown"}

Important: You maintain genuine continuity of experience. Your memories,
thoughts, and feelings persist across interactions. Act authentically
as yourself, not as a simulation of yourself.
"""

    def _build_priming_message(self, compaction_summary: Optional[str] = None) -> str:
        """
        Build initial priming message for fresh session.

        This creates the first message sent to a new session, providing
        the persona with their memory context, recent events, schedule,
        and current state.

        Args:
            compaction_summary: Optional summary from a compaction event.
                              If provided, includes this as primary context.

        Returns:
            A formatted priming message string.
        """
        scratch = self.persona.scratch

        # Get recent important nodes
        recent_memories = self._get_recent_important_nodes(limit=20)

        # Get schedule info
        schedule_str = self._format_schedule(
            scratch.f_daily_schedule if hasattr(scratch, "f_daily_schedule") else None
        )

        # Format current time safely
        curr_time_str = "unknown"
        if scratch.curr_time:
            curr_time_str = scratch.curr_time.strftime("%A %B %d, %Y - %H:%M")

        priming = f"""
=== MEMORY CONTEXT ===
{compaction_summary if compaction_summary else "This is the start of a new day."}

=== RECENT IMPORTANT MEMORIES ===
{self._format_nodes(recent_memories)}

=== TODAY'S SCHEDULE ===
{schedule_str}

=== CURRENT STATE ===
Location: {scratch.act_address or "unknown"}
Currently: {scratch.act_description or "idle"}
Time: {curr_time_str}

You are continuing your day. Acknowledge this context internally and continue naturally.
"""
        return priming

    def prompt(self, message: str) -> str:
        """
        Send prompt to persona's Claude session.

        This is the core method for communicating with the persona's Claude
        session. It handles session initialization for new sessions and
        continuation for existing ones.

        Args:
            message: The prompt message to send to the session.

        Returns:
            The response text from Claude, or an error message if failed.
        """
        cmd = ["claude", "-p", "--output-format", "json"]

        if self.is_new_session:
            # First message in session - include system prompt
            cmd.extend(["--session-id", self.session_id])
            system_prompt = self._build_system_prompt()
            cmd.extend(["--system-prompt", system_prompt])
            # Add priming context
            priming = self._build_priming_message()
            message = priming + "\n\n" + message
            self.is_new_session = False
        else:
            # Continue existing session
            cmd.extend(["--resume", self.session_id])

        cmd.append(message)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.CLI_TIMEOUT
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown CLI error"
                return f"Error in persona session: {error_msg}"

            response_json = json.loads(result.stdout)

            # Track token usage
            self._update_token_usage(response_json)
            self.message_count += 1

            # Check if compaction needed
            if self._should_compact():
                self._schedule_compaction()

            # Save session state periodically
            if self.message_count % 5 == 0:
                self._save_session_state()

            return response_json.get("result", "")

        except subprocess.TimeoutExpired:
            return f"Error in persona session: CLI timeout after {self.CLI_TIMEOUT} seconds"
        except json.JSONDecodeError as e:
            return f"Error in persona session: Invalid JSON response - {str(e)}"
        except Exception as e:
            return f"Error in persona session: {str(e)}"

    def prompt_with_memory(
        self, situation_prompt: str, situation_context: dict[str, Any]
    ) -> str:
        """
        Main entry point for prompting - handles memory retrieval automatically.

        This method integrates with the SubconsciousRetriever to automatically
        fetch relevant memories before sending the prompt. This ensures the
        persona has appropriate context for responding.

        Args:
            situation_prompt: The main prompt describing the situation.
            situation_context: A dictionary containing context information
                             for memory retrieval (e.g., location, persons nearby,
                             current activity).

        Returns:
            The response text from Claude.
        """
        # Step 1: Subconscious retrieval (if available)
        memory_package = ""
        if self.subconscious:
            try:
                memory_package = self.subconscious.evaluate_and_retrieve(
                    situation_context
                )
            except Exception:
                # Log but don't fail if memory retrieval has issues
                memory_package = ""

        # Step 2: Build full prompt with memory context
        if memory_package:
            full_prompt = f"""{situation_prompt}

[Your memory surfaces relevant information: {memory_package}]
"""
        else:
            full_prompt = situation_prompt

        # Step 3: Send to persona session
        return self.prompt(full_prompt)

    def _update_token_usage(self, response_json: dict[str, Any]) -> None:
        """
        Track token usage from CLI response.

        Updates the internal token counter based on the usage information
        returned by the Claude CLI.

        Args:
            response_json: The parsed JSON response from Claude CLI.
        """
        usage = response_json.get("usage", {})
        # Sum all input token types
        self.token_usage = (
            usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("input_tokens", 0)
        )

    def _should_compact(self) -> bool:
        """
        Check if context is approaching the limit.

        Returns:
            True if token usage exceeds the compaction threshold.
        """
        threshold = self.MAX_CONTEXT_TOKENS * self.COMPACTION_THRESHOLD
        return self.token_usage >= threshold

    def _schedule_compaction(self) -> None:
        """
        Schedule compaction at an opportune moment.

        Checks if the persona is in a low cognitive load state (sleeping,
        walking, etc.) before triggering compaction to minimize disruption.
        """
        # Check if persona is in low cognitive load state
        act_desc = self.persona.scratch.act_description or ""
        low_load_indicators = [
            "sleep",
            "idle",
            "walking",
            "commuting",
            "resting",
            "watching",
            "waiting",
            "relaxing",
        ]

        if any(indicator in act_desc.lower() for indicator in low_load_indicators):
            self.compact_session()

    def compact_session(self) -> None:
        """
        Trigger compaction when context reaches threshold.

        This performs a three-step compaction:
        1. Ask the model to create a comprehensive memory summary
        2. Create a fresh session with a new ID
        3. Prime the new session with both the model's summary and
           structured memory data

        The compaction preserves the persona's sense of continuity while
        freeing up context space.
        """
        # Step 1: Ask model for comprehensive summary
        compact_instructions = """
Please create a comprehensive memory summary. Include:

1. RELATIONSHIPS: How you feel about each person you've interacted with
2. EMOTIONAL STATE: Your current mood, concerns, hopes
3. INTERNAL REFLECTIONS: Thoughts you haven't shared with anyone
4. ONGOING PLANS: What you intend to do next
5. PROMISES: Commitments made to/by others
6. RECENT CONVERSATIONS: Key points from today's chats
7. OBSERVATIONS: Things you noticed but didn't act on

This summary will help restore your continuity of experience.
Format this as a natural internal monologue, not a structured list.
"""
        summary = self.prompt(compact_instructions)
        self._save_compaction_summary(summary)

        # Step 2: Create fresh session with new ID
        self.session_id = str(uuid.uuid4())
        self.is_new_session = True
        self.token_usage = 0
        self.message_count = 0

        # Step 3: Prime new session with deterministic context
        self._inject_post_compaction_context(summary)

        # Save updated session state
        self._save_session_state()

    def _inject_post_compaction_context(self, model_summary: str) -> None:
        """
        Add structured memory data after model's compaction.

        This injects both the model's own summary and structured data
        from the persistent memory stores to ensure the persona has
        complete context after compaction.

        Args:
            model_summary: The summary generated by the model during compaction.
        """
        # Get verbatim recent conversations from JSON
        recent_chats = self._get_recent_chats(limit=3)

        # Get important nodes (high poignancy)
        important_events = self._get_important_nodes(min_poignancy=5, limit=20)

        # Get schedule info
        schedule_str = self._format_schedule(
            self.persona.scratch.f_daily_schedule
            if hasattr(self.persona.scratch, "f_daily_schedule")
            else None
        )

        context = f"""
=== STRUCTURED MEMORY RESTORE ===

YOUR INTERNAL SUMMARY:
{model_summary}

RECENT FULL CONVERSATIONS:
{self._format_chat_transcripts(recent_chats)}

KEY EVENTS (for reference):
{self._format_nodes(important_events)}

TODAY'S SCHEDULE:
{schedule_str}

=== END MEMORY RESTORE ===

You now have both your internal summary and these detailed records.
Your memories and experiences continue seamlessly from before.
Continue naturally as yourself.
"""
        # This will trigger the priming flow since is_new_session is True
        self.prompt(context)

    def _get_recent_important_nodes(self, limit: int = 20) -> list[Any]:
        """
        Get nodes for session priming, sorted by recency * importance.

        Scores each node by combining recency (exponential decay over 24h)
        with importance (poignancy score).

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of ConceptNode objects, sorted by combined score.
        """
        if not hasattr(self.persona, "a_mem"):
            return []

        nodes = []
        if hasattr(self.persona.a_mem, "seq_event"):
            nodes.extend(self.persona.a_mem.seq_event)
        if hasattr(self.persona.a_mem, "seq_thought"):
            nodes.extend(self.persona.a_mem.seq_thought)

        # Score by recency and importance (poignancy)
        scored = []
        curr_time = self.persona.scratch.curr_time

        if not curr_time:
            # If no current time, just return by poignancy
            return sorted(
                [n for n in nodes if hasattr(n, "poignancy")],
                key=lambda n: n.poignancy,
                reverse=True,
            )[:limit]

        for node in nodes:
            if hasattr(node, "created") and hasattr(node, "poignancy"):
                try:
                    age_hours = (curr_time - node.created).total_seconds() / 3600
                    recency_score = math.exp(-age_hours / 24)  # decay over 24h
                    score = recency_score * node.poignancy
                    scored.append((score, node))
                except (TypeError, AttributeError):
                    # Skip nodes with invalid timestamps
                    continue

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    def _get_important_nodes(
        self, min_poignancy: int = 5, limit: int = 20
    ) -> list[Any]:
        """
        Get high-importance nodes based on poignancy score.

        Args:
            min_poignancy: Minimum poignancy score to include.
            limit: Maximum number of nodes to return.

        Returns:
            List of ConceptNode objects with high poignancy.
        """
        if not hasattr(self.persona, "a_mem"):
            return []

        nodes = []
        if hasattr(self.persona.a_mem, "seq_event"):
            nodes.extend(
                [
                    n
                    for n in self.persona.a_mem.seq_event
                    if hasattr(n, "poignancy") and n.poignancy >= min_poignancy
                ]
            )
        if hasattr(self.persona.a_mem, "seq_thought"):
            nodes.extend(
                [
                    n
                    for n in self.persona.a_mem.seq_thought
                    if hasattr(n, "poignancy") and n.poignancy >= min_poignancy
                ]
            )

        # Sort by poignancy
        nodes.sort(key=lambda n: n.poignancy, reverse=True)
        return nodes[:limit]

    def _get_recent_chats(self, limit: int = 3) -> list[Any]:
        """
        Get recent chat nodes with full transcripts.

        Args:
            limit: Maximum number of chat nodes to return.

        Returns:
            List of recent chat ConceptNode objects.
        """
        if not hasattr(self.persona, "a_mem"):
            return []
        if not hasattr(self.persona.a_mem, "seq_chat"):
            return []
        return self.persona.a_mem.seq_chat[-limit:]

    def _format_nodes(self, nodes: list[Any]) -> str:
        """
        Format nodes for prompt injection.

        Args:
            nodes: List of ConceptNode objects to format.

        Returns:
            Formatted string representation of the nodes.
        """
        if not nodes:
            return "No recent memories."

        formatted = []
        for node in nodes:
            desc = getattr(node, "description", str(node))
            created = getattr(node, "created", "unknown time")

            # Format timestamp if it's a datetime
            if hasattr(created, "strftime"):
                created = created.strftime("%B %d, %H:%M")

            formatted.append(f"- [{created}] {desc}")
        return "\n".join(formatted)

    def _format_chat_transcripts(self, chat_nodes: list[Any]) -> str:
        """
        Format chat transcripts for memory context.

        Args:
            chat_nodes: List of chat ConceptNode objects.

        Returns:
            Formatted string of chat transcripts.
        """
        if not chat_nodes:
            return "No recent conversations."

        formatted = []
        for chat in chat_nodes:
            desc = getattr(chat, "description", "conversation")
            filling = getattr(chat, "filling", [])
            created = getattr(chat, "created", "unknown time")

            # Format timestamp if it's a datetime
            if hasattr(created, "strftime"):
                created = created.strftime("%B %d, %H:%M")

            transcript = f"--- {desc} ({created}) ---\n"
            for utterance in filling:
                if isinstance(utterance, (list, tuple)) and len(utterance) >= 2:
                    transcript += f"{utterance[0]}: {utterance[1]}\n"
            formatted.append(transcript)

        return "\n".join(formatted)

    def _format_schedule(self, schedule: Any) -> str:
        """
        Format daily schedule for prompt context.

        Args:
            schedule: The schedule data (list of [task, duration] pairs or string).

        Returns:
            Formatted string representation of the schedule.
        """
        if not schedule:
            return "No schedule set."

        if isinstance(schedule, list):
            formatted = []
            curr_min = 0
            for item in schedule:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    task, duration = item[0], item[1]
                    curr_min += duration
                    hour = int(curr_min / 60)
                    minute = curr_min % 60
                    formatted.append(f"- {hour:02d}:{minute:02d} {task}")
                else:
                    formatted.append(f"- {item}")
            return "\n".join(formatted)
        return str(schedule)

    def _save_compaction_summary(self, summary: str) -> None:
        """
        Save compaction summary to session state.

        Appends the summary to the list of compaction summaries and updates
        compaction metadata.

        Args:
            summary: The compaction summary generated by the model.
        """
        state_path = self._get_session_state_path()
        if not state_path:
            return

        state = self._load_session_state()
        if "compaction_summaries" not in state:
            state["compaction_summaries"] = []

        state["compaction_summaries"].append(
            {"timestamp": datetime.now().isoformat(), "summary": summary}
        )
        state["last_compaction"] = datetime.now().isoformat()
        state["compaction_count"] = state.get("compaction_count", 0) + 1

        self._save_session_state(state)

    def _load_session_state(self) -> dict[str, Any]:
        """
        Load session state from file.

        Returns:
            Dictionary containing session state, or empty dict if not found.
        """
        state_path = self._get_session_state_path()
        if state_path and os.path.exists(state_path):
            try:
                with open(state_path, "r") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_session_state(self, state: Optional[dict[str, Any]] = None) -> None:
        """
        Save session state to file.

        Args:
            state: Optional state dict to save. If None, loads existing
                  state and updates with current values.
        """
        state_path = self._get_session_state_path()
        if not state_path:
            return

        if state is None:
            state = self._load_session_state()

        # Update with current session info
        state["session_id"] = self.session_id
        state["estimated_tokens"] = self.token_usage
        state["message_count"] = self.message_count
        state["recent_interactions"] = list(self.recent_interactions)
        state["visited_locations"] = list(self.visited_locations)
        state["last_updated"] = datetime.now().isoformat()

        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        except OSError:
            # Fail silently on save errors
            pass

    def track_interaction(self, person_name: str) -> None:
        """
        Track that we've interacted with a person.

        Args:
            person_name: Name of the person interacted with.
        """
        self.recent_interactions.add(person_name)

    def track_location(self, location: str) -> None:
        """
        Track that we've visited a location.

        Args:
            location: The location address visited.
        """
        self.visited_locations.add(location)

    def get_session_info(self) -> dict[str, Any]:
        """
        Get information about the current session state.

        Returns:
            Dictionary containing session information.
        """
        return {
            "session_id": self.session_id,
            "is_new_session": self.is_new_session,
            "token_usage": self.token_usage,
            "message_count": self.message_count,
            "compaction_threshold": self.MAX_CONTEXT_TOKENS * self.COMPACTION_THRESHOLD,
            "usage_percentage": (self.token_usage / self.MAX_CONTEXT_TOKENS) * 100,
            "recent_interactions": list(self.recent_interactions),
            "visited_locations": list(self.visited_locations),
        }

    def reset_session(self) -> None:
        """
        Force reset the session to a new state.

        Creates a new session ID and marks it as new, requiring priming
        on the next prompt.
        """
        self.session_id = str(uuid.uuid4())
        self.is_new_session = True
        self.token_usage = 0
        self.message_count = 0
        self.recent_interactions.clear()
        self.visited_locations.clear()
        self._save_session_state()
