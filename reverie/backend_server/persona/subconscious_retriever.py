"""
SubconsciousRetriever - Proactive memory retrieval for Claudeville personas

This replaces embedding-based cosine similarity with semantic Sonnet-powered
retrieval. The "subconscious" decides when memories are needed and surfaces
them naturally, like human recall.

PRIVACY: Only accesses this persona's memories. No tool access.

Architecture:
    - Called BEFORE each persona prompt (not by persona)
    - Fresh stateless Sonnet call each time
    - Returns summarized memory package, not raw data
    - Persona receives natural language memories injected into context
"""

import json
import subprocess
from typing import Any, Optional

# System prompt for the subconscious retriever
SUBCONSCIOUS_SYSTEM_PROMPT = """You are the SUBCONSCIOUS MEMORY SYSTEM for {persona_name}.

Your role is to act as {persona_name}'s long-term memory, retrieving and
SUMMARIZING relevant memories for the current situation. You are like the
subconscious part of their mind that surfaces relevant memories without
conscious effort.

=== YOUR TASK ===

1. ANALYZE the current situation
2. SEARCH the memory database for relevant memories
3. SUMMARIZE relevant memories CONCISELY (preserve persona's context window!)
4. Return a memory package the persona can use

=== OUTPUT FORMAT (strict JSON) ===
{{
  "situation_analysis": "Brief description of what triggered memory search",
  "relevant_memories": [
    {{
      "type": "event|thought|chat",
      "date": "...",
      "summary": "CONCISE summary (not full transcript!)",
      "key_details": ["specific detail 1", "specific detail 2"],
      "emotional_tone": "positive|negative|neutral"
    }}
  ],
  "memory_package": "Natural language summary to inject into persona's context.
    Write this as if surfacing to {persona_name}'s conscious mind.
    Example: 'You remember that Maria offered to help with decorations
    yesterday. She seemed enthusiastic about the party.'"
}}

=== CRITICAL RULES ===

1. SUMMARIZE, don't dump raw data. Full conversation transcripts waste tokens.
2. Include KEY DETAILS that might be important (names, dates, promises, numbers)
3. If conversation content is needed, extract relevant quotes only
4. Write memory_package in second person ("You remember...")
5. Maximum 200 words for memory_package
6. If no relevant memories, return {{"relevant_memories": [], "memory_package": ""}}
7. You can ONLY use memories from the provided database - nothing else
8. NEVER reference other personas' private thoughts or memories
"""


class SubconsciousRetriever:
    """
    Proactive memory retrieval - runs BEFORE persona prompts.

    This class implements a "subconscious" memory system that automatically
    retrieves and summarizes relevant memories before each persona interaction.
    Unlike traditional embedding-based retrieval, this uses a Sonnet model to
    semantically understand what memories are relevant.

    Key features:
        - Privacy-first: Only accesses this persona's memories
        - Sandboxed: No tool access (--allowedTools "")
        - Stateless: Fresh call each time, no persistence
        - Efficient: Quick heuristics before invoking model

    Attributes:
        persona: The persona object this retriever serves
        recent_interactions: Set of person names seen this session
        visited_locations: Set of locations visited this session
    """

    def __init__(self, persona) -> None:
        """
        Initialize the subconscious retriever for a specific persona.

        Args:
            persona: The persona object containing memory (a_mem) and
                     scratch pad (scratch) with current state
        """
        self.persona = persona
        self.recent_interactions: set[str] = set()  # Persons seen this session
        self.visited_locations: set[str] = set()  # Locations visited this session

    def evaluate_and_retrieve(self, current_situation: dict[str, Any]) -> str:
        """
        Called before each persona prompt. Evaluates if memory retrieval is
        needed and returns a memory package to inject into persona's context.

        This is the main entry point for the subconscious retriever. It first
        checks if retrieval is needed using quick heuristics, then invokes
        Sonnet to search and summarize memories if necessary.

        Args:
            current_situation: Dictionary describing the current situation.
                Expected keys:
                - new_person: str or None (name of new person appearing)
                - new_location: str or None (new location entered)
                - conversation_keywords: list[str] (keywords from conversation)
                - time_skip: bool (significant time has passed)
                - current_activity: str (what persona is doing)
                - nearby_people: list[str] (people in vicinity)

        Returns:
            str: Memory package to inject into persona's context, or empty
                 string if no relevant memories found or retrieval not needed.

        Example:
            >>> situation = {
            ...     "new_person": "Maria",
            ...     "current_activity": "walking to the park"
            ... }
            >>> memory = retriever.evaluate_and_retrieve(situation)
            >>> print(memory)
            "You remember that Maria mentioned she loves the park near Oak Street.
             She said the flowers there remind her of her grandmother's garden."
        """
        # Quick check: do we even need retrieval?
        if not self._needs_retrieval(current_situation):
            return ""

        # Build the retrieval prompt
        prompt = self._build_prompt(current_situation)

        # Execute fresh Sonnet call (no persistence, no tools)
        result = self._run_retrieval(prompt)

        # Extract just the memory package
        try:
            parsed = json.loads(result)
            memory_package = parsed.get("memory_package", "")

            # Update tracking
            if current_situation.get("new_person"):
                self.recent_interactions.add(current_situation["new_person"])
            if current_situation.get("new_location"):
                self.visited_locations.add(current_situation["new_location"])

            return memory_package
        except (json.JSONDecodeError, TypeError, AttributeError):
            # If parsing fails, return empty - fail gracefully
            return ""

    def _needs_retrieval(self, situation: dict[str, Any]) -> bool:
        """
        Quick heuristic check before invoking model.

        This function provides fast filtering to avoid unnecessary Sonnet calls.
        It checks for common triggers that indicate memory retrieval would be
        valuable.

        Args:
            situation: Dictionary with current situation details

        Returns:
            bool: True if retrieval should be attempted, False otherwise
        """
        # New person not seen in current session
        new_person = situation.get("new_person")
        if new_person and new_person not in self.recent_interactions:
            return True

        # New location not visited in current session
        new_location = situation.get("new_location")
        if new_location and new_location not in self.visited_locations:
            return True

        # Significant time skip (e.g., waking up)
        if situation.get("time_skip"):
            return True

        # Conversation with potential memory triggers
        keywords = situation.get("conversation_keywords", [])
        if keywords and self._check_keyword_triggers(keywords):
            return True

        return False

    def _check_keyword_triggers(self, keywords: list[str]) -> bool:
        """
        Check if keywords match high-importance memory keywords.

        Looks for overlap between conversation keywords and keywords that
        appear frequently in the persona's memory (indicating importance).

        Args:
            keywords: List of keywords extracted from current conversation

        Returns:
            bool: True if any keyword matches important memory keywords
        """
        if not hasattr(self.persona, "a_mem"):
            return False

        # Get persona's important keywords from memory
        important_keywords: set[str] = set()

        # Check kw_to_event for high-frequency keywords (appears in 3+ events)
        if hasattr(self.persona.a_mem, "kw_to_event"):
            try:
                for kw, events in self.persona.a_mem.kw_to_event.items():
                    if len(events) >= 3:
                        important_keywords.add(kw.lower())
            except (TypeError, AttributeError):
                pass  # Handle malformed data gracefully

        # Check kw_to_thought for high-frequency keywords (appears in 2+ thoughts)
        if hasattr(self.persona.a_mem, "kw_to_thought"):
            try:
                for kw, thoughts in self.persona.a_mem.kw_to_thought.items():
                    if len(thoughts) >= 2:
                        important_keywords.add(kw.lower())
            except (TypeError, AttributeError):
                pass  # Handle malformed data gracefully

        # Check for overlap
        for kw in keywords:
            if kw.lower() in important_keywords:
                return True

        return False

    def _build_prompt(self, situation: dict[str, Any]) -> str:
        """
        Build the full prompt for retrieval.

        Constructs a detailed prompt including the current situation,
        persona's state, and their complete memory database for the
        Sonnet model to search through.

        Args:
            situation: Dictionary with current situation details

        Returns:
            str: Complete prompt for the Sonnet retrieval call
        """
        persona_name = self._get_persona_name()
        memory_context = self._build_memory_context()

        # Build situation description
        situation_parts: list[str] = []
        if situation.get("new_person"):
            situation_parts.append(f"{situation['new_person']} has appeared nearby.")
        if situation.get("new_location"):
            situation_parts.append(f"You have entered {situation['new_location']}.")
        if situation.get("time_skip"):
            situation_parts.append("Significant time has passed.")
        if situation.get("conversation_keywords"):
            keywords_str = ", ".join(situation["conversation_keywords"])
            situation_parts.append(f"The conversation mentions: {keywords_str}")
        if situation.get("current_activity"):
            situation_parts.append(f"Current activity: {situation['current_activity']}")
        if situation.get("nearby_people"):
            nearby_str = ", ".join(situation["nearby_people"])
            situation_parts.append(f"Nearby people: {nearby_str}")

        situation_desc = (
            " ".join(situation_parts) if situation_parts else "Normal situation."
        )

        # Get current state with safe fallbacks
        current_activity = self._get_scratch_attr("act_description", "idle")
        current_location = self._get_scratch_attr("act_address", "unknown")
        current_time = self._get_scratch_attr("curr_time", "unknown")

        prompt = f"""=== CURRENT SITUATION ===
{situation_desc}

=== PERSONA'S CURRENT STATE ===
Name: {persona_name}
Current activity: {current_activity}
Current location: {current_location}
Current time: {current_time}

=== LONG-TERM MEMORY DATABASE ===
(This contains ONLY {persona_name}'s memories - events they experienced,
thoughts they had, conversations they participated in)

{memory_context}

Search the memory database and return a memory package if relevant memories exist.
"""
        return prompt

    def _get_persona_name(self) -> str:
        """
        Safely get the persona's name.

        Returns:
            str: The persona's name, or "Unknown" if not available
        """
        try:
            if hasattr(self.persona, "scratch") and hasattr(
                self.persona.scratch, "name"
            ):
                return self.persona.scratch.name or "Unknown"
        except AttributeError:
            pass
        return "Unknown"

    def _get_scratch_attr(self, attr: str, default: str) -> str:
        """
        Safely get an attribute from the persona's scratch pad.

        Args:
            attr: Attribute name to retrieve
            default: Default value if attribute doesn't exist

        Returns:
            str: The attribute value or default
        """
        try:
            if hasattr(self.persona, "scratch"):
                value = getattr(self.persona.scratch, attr, None)
                if value is not None:
                    return str(value)
        except AttributeError:
            pass
        return default

    def _build_memory_context(self) -> str:
        """
        Build ONLY this persona's memories - privacy boundary.

        This is a critical privacy function. It extracts memories from the
        persona's associative memory (a_mem) and formats them for the
        retrieval model. Only this persona's memories are included.

        Returns:
            str: JSON-formatted string of persona's memories, or a message
                 indicating no memories are available
        """
        memories: list[dict[str, Any]] = []

        if not hasattr(self.persona, "a_mem"):
            return "No memories available."

        # This persona's events only (limit to recent/important)
        if hasattr(self.persona.a_mem, "seq_event"):
            try:
                events = self.persona.a_mem.seq_event
                if events:
                    for node in events[-100:]:  # Last 100 events
                        memory_entry = self._extract_memory_node(node, "event")
                        if memory_entry:
                            memories.append(memory_entry)
            except (TypeError, AttributeError):
                pass  # Handle malformed data gracefully

        # This persona's thoughts only
        if hasattr(self.persona.a_mem, "seq_thought"):
            try:
                thoughts = self.persona.a_mem.seq_thought
                if thoughts:
                    for node in thoughts[-50:]:  # Last 50 thoughts
                        memory_entry = self._extract_memory_node(node, "thought")
                        if memory_entry:
                            memories.append(memory_entry)
            except (TypeError, AttributeError):
                pass  # Handle malformed data gracefully

        # This persona's conversations (they participated in)
        if hasattr(self.persona.a_mem, "seq_chat"):
            try:
                chats = self.persona.a_mem.seq_chat
                if chats:
                    for node in chats[-30:]:  # Last 30 conversations
                        memory_entry = self._extract_chat_node(node)
                        if memory_entry:
                            memories.append(memory_entry)
            except (TypeError, AttributeError):
                pass  # Handle malformed data gracefully

        if not memories:
            return "No memories available."

        return json.dumps(memories, indent=2, default=str)

    def _extract_memory_node(
        self, node: Any, memory_type: str
    ) -> Optional[dict[str, Any]]:
        """
        Extract a memory node (event or thought) into a dictionary.

        Args:
            node: The memory node object
            memory_type: Either "event" or "thought"

        Returns:
            Optional[Dict]: Dictionary with memory details, or None if extraction fails
        """
        try:
            # Get date with safe fallback
            date_str = "unknown"
            if hasattr(node, "created"):
                try:
                    date_str = node.created.strftime("%Y-%m-%d %H:%M")
                except (AttributeError, ValueError):
                    date_str = str(node.created)

            # Get description
            description = getattr(node, "description", None)
            if description is None:
                description = str(node)

            # Get poignancy (importance score)
            poignancy = getattr(node, "poignancy", 1)

            return {
                "type": memory_type,
                "date": date_str,
                "description": description,
                "poignancy": poignancy,
            }
        except Exception:
            return None

    def _extract_chat_node(self, node: Any) -> Optional[dict[str, Any]]:
        """
        Extract a chat/conversation node into a dictionary.

        Chat nodes may contain full transcripts which the retriever
        will summarize appropriately.

        Args:
            node: The chat memory node object

        Returns:
            Optional[Dict]: Dictionary with chat details, or None if extraction fails
        """
        try:
            # Get date with safe fallback
            date_str = "unknown"
            if hasattr(node, "created"):
                try:
                    date_str = node.created.strftime("%Y-%m-%d %H:%M")
                except (AttributeError, ValueError):
                    date_str = str(node.created)

            # Get description
            description = getattr(node, "description", "conversation")

            # Get transcript (full for retriever to summarize)
            transcript = getattr(node, "filling", [])

            # Get poignancy (importance score) - chats default to higher
            poignancy = getattr(node, "poignancy", 5)

            return {
                "type": "chat",
                "date": date_str,
                "description": description,
                "transcript": transcript,
                "poignancy": poignancy,
            }
        except Exception:
            return None

    def _run_retrieval(self, prompt: str) -> str:
        """
        Fresh Sonnet call - no tools, no persistence.

        Executes a sandboxed Claude CLI call using Sonnet model for fast,
        cheap memory retrieval. The call is completely stateless and has
        no tool access for security.

        Args:
            prompt: The full retrieval prompt

        Returns:
            str: JSON response from Sonnet, or empty JSON object on failure
        """
        persona_name = self._get_persona_name()
        system_prompt = SUBCONSCIOUS_SYSTEM_PROMPT.format(persona_name=persona_name)

        # Build command with sandboxing
        # Note: --allowedTools "" disables all tool access
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            "sonnet",  # Fast, cheap, good enough for retrieval
            "--allowedTools",
            "",  # CRITICAL: No tools - completely sandboxed
            "--system-prompt",
            system_prompt,
            prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
            )

            if result.returncode != 0:
                # Log error for debugging but don't crash
                return "{}"

            # Parse the response
            response = json.loads(result.stdout)
            return response.get("result", "{}")

        except subprocess.TimeoutExpired:
            # Retrieval took too long - skip it
            return "{}"
        except json.JSONDecodeError:
            # Malformed response - skip it
            return "{}"
        except FileNotFoundError:
            # Claude CLI not installed
            return "{}"
        except Exception:
            # Any other error - fail gracefully
            return "{}"

    def reset_session_tracking(self) -> None:
        """
        Reset tracking when session is compacted/refreshed.

        This should be called when the persona's main session is compacted
        or refreshed, to ensure the subconscious doesn't incorrectly skip
        retrieval for "already seen" people/places.
        """
        self.recent_interactions.clear()
        self.visited_locations.clear()

    def get_session_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current session's tracking state.

        Useful for debugging and logging purposes.

        Returns:
            Dict with tracking information
        """
        return {
            "recent_interactions": list(self.recent_interactions),
            "visited_locations": list(self.visited_locations),
            "persona_name": self._get_persona_name(),
        }
