"""
Claude Agent SDK Wrapper for Claudeville - Unified Prompting System

This module provides the unified prompting architecture for Claudeville personas.
Each persona gets a single LLM call per simulation step that returns all decisions
in a structured JSON format.

Key features:
- UnifiedPersonaClient: One call per step, all decisions batched
- Initial prompt sent once at session start and after compaction
- Step prompts contain only world updates (perceptions, time, location)
- Automatic context monitoring with 80% threshold compaction
- Full Opus agency - model decides actions naturally

Author: Claudeville Project
"""

from __future__ import annotations

import asyncio
import atexit
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import ResultMessage

import cli_interface as cli

if TYPE_CHECKING:
    from persona.persona import Persona

# ============================================================================
# #####################[SECTION 1: CONFIGURATION] ############################
# ============================================================================

# Context window limits
MAX_CONTEXT_TOKENS = 200000  # Claude Opus context window
COMPACTION_THRESHOLD = 0.80  # Trigger compaction at 80% fill
COMPACTION_TOKEN_LIMIT = int(MAX_CONTEXT_TOKENS * COMPACTION_THRESHOLD)  # 160K tokens

# Debug verbosity (0=silent, 1=summary, 2=decisions, 3=full prompts)
DEBUG_VERBOSITY = 1

# Track last printed action per persona (to avoid duplicate output)
# Format: {persona_name: action_description}
_last_printed_action: dict[str, str] = {}


# ============================================================================
# #####################[SECTION 2: DATA STRUCTURES] ##########################
# ============================================================================


@dataclass
class ActionDecision:
    """Parsed action decision from model response."""

    description: str
    duration_minutes: int
    sector: str
    arena: str
    game_object: str
    emoji: str
    event: tuple[str, str, str]  # (subject, predicate, object)


@dataclass
class SocialDecision:
    """Parsed social decision from model response."""

    wants_to_talk: bool = False
    target: str | None = None
    conversation_line: str | None = None
    continue_conversation: bool = False


@dataclass
class ThoughtDecision:
    """Parsed thought/reflection from model response."""

    content: str
    importance: int = 5


@dataclass
class StepResponse:
    """Fully parsed response from a step prompt."""

    action: ActionDecision | None = None
    social: SocialDecision = field(default_factory=SocialDecision)
    thoughts: list[ThoughtDecision] = field(default_factory=list)
    schedule_update: list[tuple[str, int]] | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)


# ============================================================================
# #####################[SECTION 3: ASYNC INFRASTRUCTURE] #####################
# ============================================================================

# Persistent event loop running in a background thread
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()

# Per-persona client pool
_persona_clients: dict[str, ClaudeSDKClient] = {}
_persona_locks: dict[str, asyncio.Lock] = {}
_persona_usage: dict[str, dict[str, Any]] = {}
_persona_initialized: dict[str, bool] = {}
_persona_colors: dict[str, str] = {}  # Assigned colors per persona

# Available colors for personas (assigned in order of registration)
PERSONA_COLORS = [
    cli.Colors.BRIGHT_CYAN,
    cli.Colors.BRIGHT_MAGENTA,
    cli.Colors.BRIGHT_YELLOW,
    cli.Colors.BRIGHT_GREEN,
    cli.Colors.BRIGHT_BLUE,
    cli.Colors.BRIGHT_RED,
    cli.Colors.BRIGHT_WHITE,
]


def get_persona_color(persona_name: str) -> str:
    """Get a unique color for a persona, assigning one if not yet assigned."""
    if persona_name not in _persona_colors:
        # Assign next available color
        used_count = len(_persona_colors)
        _persona_colors[persona_name] = PERSONA_COLORS[used_count % len(PERSONA_COLORS)]
    return _persona_colors[persona_name]


def _get_or_start_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop running in a background thread."""
    global _loop, _loop_thread

    with _loop_lock:
        if _loop is None or not _loop.is_running():
            _loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(_loop)
                _loop.run_forever()

            _loop_thread = threading.Thread(target=run_loop, daemon=True)
            _loop_thread.start()
            atexit.register(_shutdown_loop)

    return _loop


def _shutdown_loop():
    """Shutdown the background event loop."""
    global _loop, _loop_thread

    if _loop is not None and _loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_cleanup_all_clients(), _loop)
        try:
            future.result(timeout=5.0)
        except Exception:
            pass
        _loop.call_soon_threadsafe(_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=2.0)

    _loop = None
    _loop_thread = None


def _run_async(coro):
    """Run an async coroutine from sync code using the persistent event loop."""
    loop = _get_or_start_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


async def _cleanup_all_clients():
    """Cleanup all persona clients."""
    for name, client in list(_persona_clients.items()):
        try:
            await client.disconnect()
        except Exception:
            pass
    _persona_clients.clear()
    _persona_usage.clear()
    _persona_initialized.clear()
    _persona_colors.clear()


# ============================================================================
# #####################[SECTION 4: PROMPT BUILDERS] ##########################
# ============================================================================


def build_initial_prompt(
    persona: Persona, compaction_summary: str | None = None
) -> str:
    """
    Build the initial prompt sent at session start or after compaction.

    This establishes the persona's identity and world context.
    Only sent once per session.
    """
    scratch = persona.scratch

    # Core identity
    name = scratch.name or "Unknown"
    age = scratch.age or "unknown age"
    innate = scratch.innate or "no defined traits"
    learned = scratch.learned or "no background"
    lifestyle = scratch.lifestyle or "no defined lifestyle"
    living_area = scratch.living_area or "unknown location"
    currently = scratch.currently or "nothing in particular"

    # Memory context
    memory_section = ""
    if compaction_summary:
        memory_section = f"""
=== YOUR MEMORIES ===
{compaction_summary}
"""
    else:
        # Get recent important memories for fresh session
        memories = _get_recent_memories(persona, limit=15)
        if memories:
            memory_section = f"""
=== RECENT MEMORIES ===
{memories}
"""

    return f"""You are {name}, a {age}-year-old living in Smallville.

=== WHO YOU ARE ===
Core traits: {innate}
Background: {learned}
Lifestyle: {lifestyle}
Home: {living_area}
Current focus: {currently}

=== THE WORLD ===
You live in a small town called Smallville. Time passes naturally. You interact
with neighbors, maintain daily routines, and make your own decisions about how
to spend your time. This is your life - act naturally as yourself.
{memory_section}
=== HOW TO RESPOND ===
When I describe what's happening around you, respond with a JSON object containing
your decisions. The required format is:

```json
{{
  "action": {{
    "description": "what you are doing",
    "duration_minutes": 30,
    "location": {{
      "sector": "exact sector name from options",
      "arena": "exact arena name from options",
      "object": "exact object name from options"
    }},
    "emoji": "1-3 emoji representing your action",
    "event": ["{name}", "verb", "object of action"]
  }},
  "social": {{
    "wants_to_talk": false,
    "target": null,
    "conversation_line": null
  }},
  "thoughts": []
}}
```

Required fields: action (with all subfields), social
Optional fields: thoughts (list of {{"content": "...", "importance": 1-10}}), schedule_update

=== CONVERSATIONS ===
When interacting with someone nearby:
- Set wants_to_talk: true
- Set target: their name
- Set conversation_line: what you actually SAY to them (dialogue in quotes)
Example: "conversation_line": "Hey Maria! How's your studying going?"
The conversation_line is ACTUAL DIALOGUE that will be shown as a speech bubble.

=== REALITY RULES ===
1. PHYSICAL: You can only interact with objects at your current location. To use something elsewhere, travel there first.
2. TEMPORAL: Actions take realistic time. A shower is 5-10 minutes, not 30. Breakfast is 15-20 minutes. Adjust duration_minutes accordingly.
3. CONTINUITY: If you're in the middle of something and nothing has changed, continue it. Don't jump between activities erratically.
4. CONSISTENCY: Your schedule is a guide, not a script. Adapt naturally to what's happening around you.

=== EVENT TRIPLE FORMAT ===
The "event" field describes your action as [subject, verb, object]:
- Subject: Always your name ("{name}")
- Verb: Simple present tense (brew, eat, read, write, sleep, walk, work)
- Object: What you're acting on (coffee, breakfast, book, document, etc.)
Examples: ["{name}", "brew", "coffee"], ["{name}", "eat", "breakfast"], ["{name}", "read", "book"]

IMPORTANT: Use EXACT location names from the options I provide. Respond with ONLY the JSON, no other text.
"""


def build_step_prompt(
    persona: Persona,
    perceptions: list[str],
    nearby_personas: list[tuple[str, str]],  # [(name, activity), ...]
    accessible_locations: dict[str, Any],  # {sector: {arena: [objects]}}
    conversation_context: list[tuple[str, str]] | None = None,  # [(speaker, line), ...]
    nearby_conversations: list[dict] | None = None,  # [{participants, chat, group_id}]
) -> str:
    """
    Build a step prompt with minimal world updates.

    This is the main prompt sent each simulation step (when needed).
    """
    scratch = persona.scratch

    # Current time
    time_str = "unknown"
    if scratch.curr_time:
        time_str = scratch.curr_time.strftime("%A %B %d, %H:%M")

    # Current location - extract sector and arena for clarity
    location = scratch.act_address or "unknown"
    location_parts = location.split(":") if location != "unknown" else []
    current_sector = location_parts[1] if len(location_parts) > 1 else "unknown"
    current_arena = location_parts[2] if len(location_parts) > 2 else "unknown"

    # Current activity with duration info
    current_action = scratch.act_description or "idle"
    action_context = ""
    if scratch.act_start_time and scratch.act_duration:
        elapsed = (scratch.curr_time - scratch.act_start_time).total_seconds() / 60
        remaining = scratch.act_duration - elapsed
        if remaining > 0:
            action_context = (
                f" (started {int(elapsed)} min ago, {int(remaining)} min remaining)"
            )
        else:
            action_context = f" (completed - {int(elapsed)} min elapsed, was planned for {scratch.act_duration} min)"

    # Format perceptions
    if perceptions:
        perception_str = "\n".join(f"- {p}" for p in perceptions)
    else:
        perception_str = "(nothing new)"

    # Format nearby personas
    if nearby_personas:
        nearby_str = "\n".join(
            f"- {name}: {activity}" for name, activity in nearby_personas
        )
    else:
        nearby_str = "(no one nearby)"

    # Format accessible locations
    location_lines = []
    for sector, arenas in accessible_locations.items():
        arena_list = []
        for arena, objects in arenas.items():
            obj_str = ", ".join(objects) if objects else "no objects"
            arena_list.append(f"{arena} ({obj_str})")
        location_lines.append(f"  {sector}:")
        for arena_line in arena_list:
            location_lines.append(f"    - {arena_line}")
    location_str = "\n".join(location_lines) if location_lines else "(none available)"

    # Format schedule (remaining items for today)
    schedule_str = _format_remaining_schedule(scratch)

    # Conversation context if active
    convo_section = ""
    positioning_guidance = ""
    if conversation_context:
        convo_lines = "\n".join(
            f"{speaker}: {line}" for speaker, line in conversation_context
        )
        convo_section = f"""
=== ACTIVE CONVERSATION ===
{convo_lines}

(It's your turn to respond. Set continue_conversation to false to end.)
"""
        # Add positioning guidance for active conversation
        positioning_guidance = """
=== CONVERSATION POSITIONING ===
You are in an active conversation. Consider your physical positioning:
- For casual chat: stay stationary, face your conversation partner(s)
- For intimate/important topics: move closer (1-2 tiles) if too far
- For greetings from afar: you may speak first, then approach
- For lectures/presentations: speaker may pace, listeners stay seated
- Stay in place while talking - don't walk away mid-conversation!

Your location should generally stay the same during conversation unless moving closer.
"""

    # Nearby conversations they could join
    nearby_convo_section = ""
    if nearby_conversations and not conversation_context:
        # Only show if not already in a conversation
        convo_strs = []
        for conv in nearby_conversations[:2]:  # Limit to 2 conversations
            participants = ", ".join(conv.get("participants", []))
            chat_preview = conv.get("chat", [])[-3:]  # Last 3 lines
            chat_lines = "\n    ".join(f"{s}: {line}" for s, line in chat_preview)
            convo_strs.append(f"  {participants}:\n    {chat_lines}")

        if convo_strs:
            nearby_convo_section = f"""
=== NEARBY CONVERSATION ===
{chr(10).join(convo_strs)}

You can hear this conversation. If socially appropriate, you may join by:
- Setting wants_to_talk: true and target to one of the participants
- Adding a natural entry line that acknowledges the ongoing discussion
- Or continue your current activity if joining wouldn't be appropriate
"""

    # Build decision guidance based on context
    decision_guidance = ""
    if nearby_personas:
        decision_guidance = """
=== DECISION ===
Someone is nearby! You may:
1. Continue your current activity (if it makes sense)
2. Greet or interact with them (set wants_to_talk: true)
3. Change what you're doing based on the social situation"""
    elif scratch.act_duration and scratch.act_start_time:
        elapsed = (scratch.curr_time - scratch.act_start_time).total_seconds() / 60
        remaining = scratch.act_duration - elapsed
        if remaining > 0:
            decision_guidance = f"""
=== DECISION ===
You are currently: {current_action}
Time remaining: {int(remaining)} minutes
Unless something important changed, you should CONTINUE your current activity.
Only change if you have a compelling reason (someone to talk to, urgent need, etc.)."""

    return f"""TIME: {time_str}
CURRENT LOCATION: {current_sector} > {current_arena}
CURRENT ACTIVITY: {current_action}{action_context}

=== PERCEPTIONS ===
{perception_str}

=== NEARBY PEOPLE ===
{nearby_str}

=== ACCESSIBLE LOCATIONS ===
{location_str}

=== YOUR SCHEDULE FOR TODAY ===
{schedule_str}
{convo_section}{nearby_convo_section}{positioning_guidance}{decision_guidance}

=== REALITY CONSTRAINTS ===
PHYSICAL:
- You are physically at: {current_sector} > {current_arena}
- You can ONLY interact with objects HERE, not elsewhere
- To use something at another location, you must TRAVEL there first
- Set your action's location to where you CURRENTLY ARE or where you're GOING

TEMPORAL:
- Current time: {time_str}
- Actions take realistic time (breakfast: 15-20min, shower: 5-10min, commute: varies by distance)
- Don't plan activities inappropriate for the time (e.g., lunch at 7am, sleeping at noon)
- Your schedule is a GUIDE, not a script - adapt to circumstances

DURATION:
- Set realistic duration_minutes for your action
- Short: greeting (1-2min), checking phone (2-3min)
- Medium: meal (15-30min), shower (5-10min), getting dressed (5-10min)
- Long: work session (60-180min), socializing (30-60min), sleeping (360-480min)

Respond with JSON only."""


def build_retry_prompt(original_prompt: str, errors: list[str]) -> str:
    """Build a retry prompt when JSON parsing fails."""
    error_list = "\n".join(f"- {e}" for e in errors)
    return f"""Your previous response had issues:
{error_list}

Please respond again with ONLY valid JSON. Use EXACT location names from the options.

{original_prompt}"""


def build_day_planning_prompt(persona: Persona, date_str: str) -> str:
    """
    Build a prompt for daily planning - wake up time and schedule.

    Called once per simulation day to generate personalized schedule.
    """
    scratch = persona.scratch

    # Core identity
    name = scratch.name or "Unknown"
    age = scratch.age or "unknown age"
    innate = scratch.innate or "no defined traits"
    learned = scratch.learned or "no background"
    lifestyle = scratch.lifestyle or "no defined lifestyle"
    living_area = scratch.living_area or "unknown location"
    currently = scratch.currently or "nothing in particular"

    return f"""You are {name}, a {age}-year-old.

=== WHO YOU ARE ===
Core traits: {innate}
Background: {learned}
Lifestyle: {lifestyle}
Home: {living_area}
Current focus: {currently}

=== TODAY ===
Today is {date_str}. Plan your day.

Based on your lifestyle and personality, decide:
1. What time do you wake up? (Consider: are you an early bird or night owl?)
2. What are your main goals for today? (as many as make sense for you)
3. What is your hourly schedule?

Respond with JSON only:
```json
{{
  "wake_up_hour": 7,
  "daily_goals": [
    "your goals here - include as many as you have"
  ],
  "schedule": [
    {{"activity": "sleeping", "duration_minutes": 420}},
    {{"activity": "wake up and morning routine", "duration_minutes": 60}},
    {{"activity": "have breakfast", "duration_minutes": 30}},
    {{"activity": "work on main task", "duration_minutes": 180}},
    {{"activity": "lunch break", "duration_minutes": 60}},
    {{"activity": "afternoon activities", "duration_minutes": 180}},
    {{"activity": "dinner", "duration_minutes": 60}},
    {{"activity": "evening relaxation", "duration_minutes": 120}},
    {{"activity": "prepare for bed", "duration_minutes": 30}},
    {{"activity": "sleeping", "duration_minutes": 300}}
  ]
}}
```

IMPORTANT:
- Schedule activities should add up to 1440 minutes (24 hours)
- Start with sleeping until your wake_up_hour
- Be specific about activities based on your personality and goals
- Respond with ONLY the JSON, no other text."""


@dataclass
class DayPlanResponse:
    """Parsed response from day planning prompt."""

    wake_up_hour: int = 7
    daily_goals: list[str] = field(default_factory=list)
    schedule: list[tuple[str, int]] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)


def parse_day_planning_response(response_text: str) -> DayPlanResponse:
    """Parse the JSON response from day planning prompt."""
    result = DayPlanResponse()

    # Try to extract JSON from response
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        result.parse_errors.append("No JSON object found in response")
        return result

    try:
        data = json.loads(json_match.group())
        result.raw_json = data
    except json.JSONDecodeError as e:
        result.parse_errors.append(f"Invalid JSON: {e}")
        return result

    # Parse wake up hour
    result.wake_up_hour = data.get("wake_up_hour", 7)
    if not isinstance(result.wake_up_hour, int):
        try:
            result.wake_up_hour = int(result.wake_up_hour)
        except (ValueError, TypeError):
            result.wake_up_hour = 7

    # Clamp to valid range
    result.wake_up_hour = max(0, min(23, result.wake_up_hour))

    # Parse daily goals
    goals = data.get("daily_goals", [])
    if isinstance(goals, list):
        result.daily_goals = [str(g) for g in goals if g]

    # Parse schedule
    schedule_data = data.get("schedule", [])
    if isinstance(schedule_data, list):
        for item in schedule_data:
            if isinstance(item, dict):
                activity = item.get("activity", "idle")
                duration = item.get("duration_minutes", 60)
                try:
                    duration = int(duration)
                except (ValueError, TypeError):
                    duration = 60
                result.schedule.append((str(activity), duration))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                result.schedule.append((str(item[0]), int(item[1])))

    # Validate total duration (should be ~1440 minutes)
    total = sum(d for _, d in result.schedule)
    if total < 1400 or total > 1480:
        result.parse_errors.append(f"Schedule total is {total} minutes, expected ~1440")

    return result


# ============================================================================
# #####################[SECTION 5: RESPONSE PARSING] #########################
# ============================================================================


def parse_step_response(
    response_text: str,
    persona_name: str,
    valid_sectors: list[str],
    valid_arenas: dict[str, list[str]],
    valid_objects: dict[str, dict[str, list[str]]],
) -> StepResponse:
    """
    Parse and validate the JSON response from a step prompt.

    Returns a StepResponse with parsed data and any parse errors.
    """
    result = StepResponse()

    # Try to extract JSON from response
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        result.parse_errors.append("No JSON object found in response")
        return result

    try:
        data = json.loads(json_match.group())
        result.raw_json = data
    except json.JSONDecodeError as e:
        result.parse_errors.append(f"Invalid JSON: {e}")
        return result

    # Parse action
    action_data = data.get("action", {})
    if not action_data:
        result.parse_errors.append("Missing 'action' field")
    else:
        location = action_data.get("location", {})
        sector = location.get("sector", "")
        arena = location.get("arena", "")
        obj = location.get("object", "")

        # Validate location (with fuzzy matching)
        if sector and sector not in valid_sectors:
            closest = _fuzzy_match(sector, valid_sectors)
            if closest:
                sector = closest
            else:
                result.parse_errors.append(f"Invalid sector: {sector}")

        if arena and sector in valid_arenas:
            if arena not in valid_arenas[sector]:
                closest = _fuzzy_match(arena, valid_arenas[sector])
                if closest:
                    arena = closest

        # Parse event triple
        event_data = action_data.get("event", [persona_name, "is", "idle"])
        if len(event_data) >= 3:
            event = (str(event_data[0]), str(event_data[1]), str(event_data[2]))
        else:
            event = (persona_name, "is", "idle")

        result.action = ActionDecision(
            description=action_data.get("description", "idle"),
            duration_minutes=action_data.get("duration_minutes", 30),
            sector=sector,
            arena=arena,
            game_object=obj,
            emoji=action_data.get("emoji", "💭"),
            event=event,
        )

    # Parse social
    social_data = data.get("social", {})
    result.social = SocialDecision(
        wants_to_talk=social_data.get("wants_to_talk", False),
        target=social_data.get("target"),
        conversation_line=social_data.get("conversation_line"),
        continue_conversation=social_data.get("continue_conversation", False),
    )

    # Parse thoughts
    thoughts_data = data.get("thoughts", [])
    for thought in thoughts_data:
        if isinstance(thought, dict):
            result.thoughts.append(
                ThoughtDecision(
                    content=thought.get("content", ""),
                    importance=thought.get("importance", 5),
                )
            )

    # Parse schedule update
    schedule_data = data.get("schedule_update")
    if schedule_data and isinstance(schedule_data, list):
        result.schedule_update = [
            (item[0], item[1]) for item in schedule_data if len(item) >= 2
        ]

    return result


def _fuzzy_match(target: str, options: list[str]) -> str | None:
    """Find closest match for target in options."""
    target_lower = target.lower().strip()
    for opt in options:
        if target_lower in opt.lower() or opt.lower() in target_lower:
            return opt
    return None


# ============================================================================
# #####################[SECTION 6: UNIFIED PERSONA CLIENT] ###################
# ============================================================================


class UnifiedPersonaClient:
    """
    Manages a persistent Claude session for a single persona.

    This is the main interface for persona prompting. Each persona gets
    one instance that maintains their session across simulation steps.
    """

    def __init__(self, persona: Persona):
        self.persona = persona
        self.persona_name = persona.name
        self._initialized = False
        self._compaction_summary: str | None = None

    async def _get_or_create_client(self) -> ClaudeSDKClient:
        """Get existing client or create new one for this persona."""
        if self.persona_name not in _persona_locks:
            _persona_locks[self.persona_name] = asyncio.Lock()

        async with _persona_locks[self.persona_name]:
            if self.persona_name not in _persona_clients:
                options = ClaudeAgentOptions(
                    allowed_tools=[],
                    permission_mode="bypassPermissions",
                    model="opus",
                )
                client = ClaudeSDKClient(options)
                await client.connect()
                _persona_clients[self.persona_name] = client
                _persona_usage[self.persona_name] = {"context_tokens": 0}
                _persona_initialized[self.persona_name] = False

            return _persona_clients[self.persona_name]

    async def _send_prompt(self, prompt: str) -> tuple[str, dict[str, Any] | None]:
        """Send a prompt and return (response_text, usage_stats)."""
        client = await self._get_or_create_client()

        await client.query(prompt)

        result_text = ""
        usage = None

        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                result_text = message.result or ""
                usage = message.usage

        # Update token tracking
        if usage:
            context_tokens = (
                usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("input_tokens", 0)
            )
            _persona_usage[self.persona_name] = {"context_tokens": context_tokens}

            # Check for compaction
            if context_tokens >= COMPACTION_TOKEN_LIMIT:
                await self._trigger_compaction()

        return result_text, usage

    async def _trigger_compaction(self):
        """Trigger context compaction."""
        if DEBUG_VERBOSITY >= 1:
            tokens = _persona_usage.get(self.persona_name, {}).get("context_tokens", 0)
            print(
                cli.c("  ⚡ ", cli.Colors.BRIGHT_YELLOW)
                + cli.c(self.persona_name, self._get_persona_color(), cli.Colors.BOLD)
                + cli.c(f" COMPACTION at {tokens:,} tokens", cli.Colors.BRIGHT_YELLOW)
            )

        # Ask model to summarize
        summary_prompt = """Please create a memory summary including:
1. How you feel about people you've met
2. Your current mood and concerns
3. What you plan to do next
4. Any promises or commitments
5. Key events from today

Write this as your internal thoughts, not a list."""

        summary, _ = await self._send_prompt(summary_prompt)
        self._compaction_summary = summary

        # Disconnect and recreate client
        if self.persona_name in _persona_clients:
            try:
                await _persona_clients[self.persona_name].disconnect()
            except Exception:
                pass
            del _persona_clients[self.persona_name]

        _persona_initialized[self.persona_name] = False

    async def _ensure_initialized(self):
        """Ensure session has received initial prompt."""
        if not _persona_initialized.get(self.persona_name, False):
            initial = build_initial_prompt(self.persona, self._compaction_summary)
            await self._send_prompt(initial)
            _persona_initialized[self.persona_name] = True
            self._compaction_summary = None  # Clear after use

            if DEBUG_VERBOSITY >= 1:
                print(
                    cli.c("  ◆ ", cli.Colors.BRIGHT_GREEN)
                    + cli.c(
                        self.persona_name, self._get_persona_color(), cli.Colors.BOLD
                    )
                    + cli.c(" session initialized", cli.Colors.DIM)
                )

    async def compact_for_sleep(self):
        """
        Trigger compaction when persona goes to sleep.

        This is called automatically when a persona starts sleeping,
        allowing the session to summarize the day's events before
        the persona wakes up with refreshed context.
        """
        if DEBUG_VERBOSITY >= 1:
            tokens = _persona_usage.get(self.persona_name, {}).get("context_tokens", 0)
            print(
                cli.c("  🌙 ", cli.Colors.BRIGHT_BLUE)
                + cli.c(self.persona_name, self._get_persona_color(), cli.Colors.BOLD)
                + cli.c(f" sleeping - compacting at {tokens:,} tokens", cli.Colors.DIM)
            )

        await self._trigger_compaction()

    async def step(
        self,
        perceptions: list[str],
        nearby_personas: list[tuple[str, str]],
        accessible_locations: dict[str, Any],
        valid_sectors: list[str],
        valid_arenas: dict[str, list[str]],
        valid_objects: dict[str, dict[str, list[str]]],
        conversation_context: list[tuple[str, str]] | None = None,
        nearby_conversations: list[dict] | None = None,
    ) -> StepResponse:
        """
        Execute a single simulation step for this persona.

        Returns a StepResponse with all parsed decisions.
        """
        await self._ensure_initialized()

        # Build step prompt
        prompt = build_step_prompt(
            self.persona,
            perceptions,
            nearby_personas,
            accessible_locations,
            conversation_context,
            nearby_conversations,
        )

        # Send and parse
        response_text, usage = await self._send_prompt(prompt)
        result = parse_step_response(
            response_text,
            self.persona_name,
            valid_sectors,
            valid_arenas,
            valid_objects,
        )

        # Retry once if parse errors
        if result.parse_errors:
            retry_prompt = build_retry_prompt(prompt, result.parse_errors)
            response_text, usage = await self._send_prompt(retry_prompt)
            result = parse_step_response(
                response_text,
                self.persona_name,
                valid_sectors,
                valid_arenas,
                valid_objects,
            )

        # Debug output using CLI colors
        self._print_step_result(result)

        return result

    async def plan_day(self, date_str: str) -> DayPlanResponse:
        """
        Generate a personalized daily schedule for the persona.

        Called once per simulation day (on new_day trigger).
        Returns wake_up_hour, daily_goals, and schedule.
        """
        await self._ensure_initialized()

        prompt = build_day_planning_prompt(self.persona, date_str)
        response_text, usage = await self._send_prompt(prompt)
        result = parse_day_planning_response(response_text)

        # Retry once if parse errors
        if result.parse_errors:
            retry_prompt = build_retry_prompt(prompt, result.parse_errors)
            response_text, usage = await self._send_prompt(retry_prompt)
            result = parse_day_planning_response(response_text)

        # Debug output
        self._print_day_plan_result(result)

        return result

    def _print_day_plan_result(self, result: DayPlanResponse):
        """Print day planning result using CLI colors."""
        if DEBUG_VERBOSITY < 1:
            return

        color = self._get_persona_color()
        tokens = _persona_usage.get(self.persona_name, {}).get("context_tokens", 0)

        name_part = cli.c(f"  📅 {self.persona_name}", color, cli.Colors.BOLD)
        tokens_part = cli.c(f" ({tokens/1000:.1f}K)", cli.Colors.DIM)

        if result.parse_errors:
            print(
                f"{name_part} day planning failed: {result.parse_errors[0]}{tokens_part}"
            )
        else:
            wake_str = f"{result.wake_up_hour}:00"
            goals_count = len(result.daily_goals)
            schedule_count = len(result.schedule)
            print(
                f"{name_part} planned day: wake {wake_str}, "
                f"{goals_count} goals, {schedule_count} activities{tokens_part}"
            )

        if DEBUG_VERBOSITY >= 2:
            for goal in result.daily_goals[:4]:
                print(cli.c(f"      Goal: {goal}", cli.Colors.DIM))
            if DEBUG_VERBOSITY >= 3:
                for activity, duration in result.schedule[:5]:
                    print(cli.c(f"      {duration}min: {activity}", cli.Colors.DIM))

    def _get_persona_color(self) -> str:
        """Get unique color for this persona."""
        return get_persona_color(self.persona_name)

    def _print_step_result(self, result: StepResponse):
        """Print step result using CLI colors. Only prints if action changed."""
        if DEBUG_VERBOSITY < 1:
            return

        # Check if action is the same as last printed
        current_action = result.action.description if result.action else "(no action)"
        last_action = _last_printed_action.get(self.persona_name)

        if current_action == last_action:
            # Action unchanged - don't print (skip logic will handle continuing output)
            return

        # Update last printed action
        _last_printed_action[self.persona_name] = current_action

        tokens = _persona_usage.get(self.persona_name, {}).get("context_tokens", 0)
        color = self._get_persona_color()

        # Time
        time_str = ""
        if self.persona.scratch.curr_time:
            time_str = self.persona.scratch.curr_time.strftime("%H:%M")

        # Build output line
        name_part = cli.c(f"  ● {self.persona_name}", color, cli.Colors.BOLD)
        time_part = cli.c(f" {time_str}", cli.Colors.DIM)

        if result.action:
            emoji = result.action.emoji
            desc = result.action.description
            tokens_part = cli.c(f" ({tokens/1000:.1f}K)", cli.Colors.DIM)
            print(f"{name_part}{time_part} {emoji} {desc}{tokens_part}")
        else:
            print(f"{name_part}{time_part} (no action)")

        # Verbose output
        if DEBUG_VERBOSITY >= 2 and result.action:
            loc = f"{result.action.sector} > {result.action.arena}"
            if result.action.game_object:
                loc += f" > {result.action.game_object}"
            print(cli.c(f"      Location: {loc}", cli.Colors.DIM))
            if result.social.wants_to_talk:
                print(
                    cli.c("      Wants to talk to: ", cli.Colors.DIM)
                    + cli.c(result.social.target, cli.Colors.BRIGHT_YELLOW)
                )

        if DEBUG_VERBOSITY >= 3:
            print(
                cli.c("      JSON: ", cli.Colors.DIM)
                + json.dumps(result.raw_json, indent=2)[:500]
            )


# ============================================================================
# #####################[SECTION 7: HELPER FUNCTIONS] #########################
# ============================================================================


def get_accessible_locations(
    persona: Persona,
) -> tuple[
    dict[str, dict[str, list[str]]],  # {sector: {arena: [objects]}}
    list[str],  # valid_sectors
    dict[str, list[str]],  # valid_arenas by sector
    dict[str, dict[str, list[str]]],  # valid_objects by sector/arena
]:
    """
    Build accessible locations dict from persona's spatial memory.

    Returns a tuple of:
    - accessible_locations: {sector: {arena: [objects]}} for prompt display
    - valid_sectors: list of all valid sector names
    - valid_arenas: {sector: [arena_names]} for validation
    - valid_objects: {sector: {arena: [object_names]}} for validation

    Example usage:
        locations, sectors, arenas, objects = get_accessible_locations(persona)
        response = client.step(perceptions, nearby, locations, sectors, arenas, objects)
    """
    accessible_locations: dict[str, dict[str, list[str]]] = {}
    valid_sectors: list[str] = []
    valid_arenas: dict[str, list[str]] = {}
    valid_objects: dict[str, dict[str, list[str]]] = {}

    if not hasattr(persona, "s_mem") or not hasattr(persona.s_mem, "tree"):
        return accessible_locations, valid_sectors, valid_arenas, valid_objects

    tree = persona.s_mem.tree

    # The tree structure is: {world: {sector: {arena: [objects]}}}
    for world_name, sectors in tree.items():
        if not isinstance(sectors, dict):
            continue

        for sector_name, arenas in sectors.items():
            if not isinstance(arenas, dict):
                continue

            # Add sector
            valid_sectors.append(sector_name)
            accessible_locations[sector_name] = {}
            valid_arenas[sector_name] = []
            valid_objects[sector_name] = {}

            for arena_name, objects in arenas.items():
                if isinstance(objects, list):
                    # Add arena
                    valid_arenas[sector_name].append(arena_name)
                    accessible_locations[sector_name][arena_name] = objects
                    valid_objects[sector_name][arena_name] = objects

    return accessible_locations, valid_sectors, valid_arenas, valid_objects


def resolve_location_to_tile(
    persona: Persona,
    maze,
    sector: str,
    arena: str,
    game_object: str | None,
) -> tuple[int, int]:
    """
    Convert sector/arena/object names to tile coordinates.

    This function resolves the JSON location decision from the LLM into actual
    tile coordinates that the persona can walk to.

    Args:
        persona: The Persona instance (used for current tile, home location)
        maze: The Maze instance (has address_tiles mapping)
        sector: Sector name from LLM decision (e.g., "Hobbs Cafe")
        arena: Arena name from LLM decision (e.g., "cafe")
        game_object: Optional game object name (e.g., "piano")

    Returns:
        Tuple (x, y) representing the target tile coordinates.
        Falls back to current tile or home if location not found.

    Example usage:
        x, y = resolve_location_to_tile(persona, maze, "Hobbs Cafe", "cafe", "piano")
    """
    # Get world name from current tile or maze
    world_name = _get_world_name(persona, maze)
    if not world_name:
        return _get_fallback_tile(persona)

    # Build address strings in order of specificity
    # Most specific (game object) to least specific (sector only)
    addresses_to_try = []

    if game_object:
        # Full address: world:sector:arena:object
        addresses_to_try.append(f"{world_name}:{sector}:{arena}:{game_object}")

    if arena:
        # Arena address: world:sector:arena
        addresses_to_try.append(f"{world_name}:{sector}:{arena}")

    if sector:
        # Sector address: world:sector
        addresses_to_try.append(f"{world_name}:{sector}")

    # Try exact matches first
    for address in addresses_to_try:
        if address in maze.address_tiles:
            tiles = maze.address_tiles[address]
            if tiles:
                # Return the first available tile
                return next(iter(tiles))

    # Try fuzzy matching on the address keys
    for address in addresses_to_try:
        matched_address = _fuzzy_match_address(address, maze.address_tiles.keys())
        if matched_address and matched_address in maze.address_tiles:
            tiles = maze.address_tiles[matched_address]
            if tiles:
                return next(iter(tiles))

    # Fall back to current tile or home
    return _get_fallback_tile(persona)


def resolve_location_to_address(
    persona: Persona,
    maze,
    sector: str,
    arena: str,
    game_object: str | None,
) -> str:
    """
    Convert sector/arena/object names to a full address string.

    Returns the address string in format "world:sector:arena:object" that can
    be used with maze.address_tiles or stored in persona.scratch.act_address.

    Args:
        persona: The Persona instance
        maze: The Maze instance
        sector: Sector name from LLM decision
        arena: Arena name from LLM decision
        game_object: Optional game object name

    Returns:
        Full address string like "the Ville:Hobbs Cafe:cafe:piano"
    """
    world_name = _get_world_name(persona, maze)
    if not world_name:
        return ""

    # Build candidate addresses from most to least specific
    addresses_to_try = []

    if game_object:
        addresses_to_try.append(f"{world_name}:{sector}:{arena}:{game_object}")

    if arena:
        addresses_to_try.append(f"{world_name}:{sector}:{arena}")

    if sector:
        addresses_to_try.append(f"{world_name}:{sector}")

    # Try exact matches
    for address in addresses_to_try:
        if address in maze.address_tiles:
            return address

    # Try fuzzy matching
    for address in addresses_to_try:
        matched = _fuzzy_match_address(address, maze.address_tiles.keys())
        if matched:
            return matched

    # Return best guess even if not in maze
    if game_object:
        return f"{world_name}:{sector}:{arena}:{game_object}"
    elif arena:
        return f"{world_name}:{sector}:{arena}"
    elif sector:
        return f"{world_name}:{sector}"
    return world_name


def _get_world_name(persona: Persona, maze) -> str | None:
    """Get the world name from persona's spatial memory or maze."""
    # Try to get from persona's spatial memory tree
    if hasattr(persona, "s_mem") and hasattr(persona.s_mem, "tree"):
        worlds = list(persona.s_mem.tree.keys())
        if worlds:
            return worlds[0]

    # Try to get from maze's tiles
    if hasattr(maze, "tiles") and maze.tiles:
        # Get world from any tile
        for row in maze.tiles:
            for tile in row:
                if tile.get("world"):
                    return tile["world"]

    return None


def _get_fallback_tile(persona: Persona) -> tuple[int, int]:
    """Get a fallback tile (current position or home spawn)."""
    # Try current tile first
    if hasattr(persona, "scratch") and hasattr(persona.scratch, "curr_tile"):
        curr = persona.scratch.curr_tile
        if curr and len(curr) >= 2:
            return (curr[0], curr[1])

    # Try living area spawn location (would need maze access)
    # For now just return a default
    return (0, 0)


def _fuzzy_match_address(target: str, options) -> str | None:
    """
    Fuzzy match an address against available options.

    Handles common issues like:
    - Case differences
    - Extra/missing spaces
    - Minor typos
    """
    target_lower = target.lower().strip()
    target_parts = target_lower.split(":")

    for opt in options:
        opt_lower = opt.lower().strip()
        opt_parts = opt_lower.split(":")

        # Exact match (case insensitive)
        if target_lower == opt_lower:
            return opt

        # Check if all parts match (fuzzy)
        if len(target_parts) == len(opt_parts):
            all_match = True
            for t_part, o_part in zip(target_parts, opt_parts):
                # Parts match if one contains the other or they're very similar
                if t_part not in o_part and o_part not in t_part:
                    # Check for high similarity (simple)
                    if not _strings_similar(t_part, o_part):
                        all_match = False
                        break
            if all_match:
                return opt

    return None


def _strings_similar(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """Check if two strings are similar enough (simple character overlap)."""
    if not s1 or not s2:
        return False

    # Simple overlap ratio
    shorter = min(len(s1), len(s2))
    if shorter == 0:
        return False

    # Count matching characters in order
    matches = 0
    s2_chars = list(s2)
    for c in s1:
        if c in s2_chars:
            s2_chars.remove(c)
            matches += 1

    return matches / max(len(s1), len(s2)) >= threshold


def find_tiles_for_location(
    maze,
    sector: str,
    arena: str | None = None,
    game_object: str | None = None,
) -> set[tuple[int, int]]:
    """
    Find all tiles matching a location query.

    Useful for getting all tiles in a sector/arena for random selection
    or finding the closest unoccupied tile.

    Args:
        maze: The Maze instance
        sector: Sector name (required)
        arena: Optional arena name
        game_object: Optional game object name

    Returns:
        Set of (x, y) tile coordinates matching the query
    """
    world_name = None
    # Get world name from maze
    if hasattr(maze, "tiles") and maze.tiles:
        for row in maze.tiles:
            for tile in row:
                if tile.get("world"):
                    world_name = tile["world"]
                    break
            if world_name:
                break

    if not world_name:
        return set()

    # Build address
    if game_object:
        address = f"{world_name}:{sector}:{arena}:{game_object}"
    elif arena:
        address = f"{world_name}:{sector}:{arena}"
    else:
        address = f"{world_name}:{sector}"

    # Try exact match
    if address in maze.address_tiles:
        return maze.address_tiles[address]

    # Try fuzzy match
    matched = _fuzzy_match_address(address, maze.address_tiles.keys())
    if matched and matched in maze.address_tiles:
        return maze.address_tiles[matched]

    return set()


def _get_recent_memories(persona: Persona, limit: int = 15) -> str:
    """Get formatted recent important memories for initial prompt."""
    if not hasattr(persona, "a_mem"):
        return ""

    lines = []

    # Get recent conversations with full content
    if hasattr(persona.a_mem, "seq_chat") and persona.a_mem.seq_chat:
        recent_chats = persona.a_mem.seq_chat[-5:]  # Last 5 conversations
        for node in recent_chats:
            created = getattr(node, "created", None)
            time_str = created.strftime("%B %d, %H:%M") if created else ""

            # Get conversation content from filling
            filling = getattr(node, "filling", None)
            if filling and isinstance(filling, list) and len(filling) > 0:
                partner = getattr(node, "object", "someone")
                lines.append(f"- [{time_str}] Conversation with {partner}:")
                for speaker, line in filling[-6:]:  # Last 6 lines of each convo
                    lines.append(f'    {speaker}: "{line}"')

    # Get recent events and thoughts
    nodes = []
    if hasattr(persona.a_mem, "seq_event"):
        nodes.extend(persona.a_mem.seq_event[-50:])
    if hasattr(persona.a_mem, "seq_thought"):
        nodes.extend(persona.a_mem.seq_thought[-20:])

    # Sort by poignancy (importance)
    nodes = [n for n in nodes if hasattr(n, "poignancy") and hasattr(n, "description")]
    nodes.sort(key=lambda n: n.poignancy, reverse=True)
    nodes = nodes[:limit]

    for node in nodes:
        desc = node.description
        created = getattr(node, "created", None)
        if created and hasattr(created, "strftime"):
            time_str = created.strftime("%B %d, %H:%M")
            lines.append(f"- [{time_str}] {desc}")
        else:
            lines.append(f"- {desc}")

    return "\n".join(lines) if lines else ""


def _format_remaining_schedule(scratch) -> str:
    """Format remaining schedule items for today."""
    if not hasattr(scratch, "f_daily_schedule") or not scratch.f_daily_schedule:
        return "(no schedule set)"

    if not scratch.curr_time:
        return "(no current time)"

    # Calculate current minute of day
    curr_min = scratch.curr_time.hour * 60 + scratch.curr_time.minute

    lines = []
    accumulated = 0
    for item in scratch.f_daily_schedule:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            task, duration = item[0], item[1]
            accumulated += duration
            if accumulated > curr_min:  # This item is in the future
                hour = accumulated // 60
                minute = accumulated % 60
                lines.append(f"- {hour:02d}:{minute:02d} {task}")

    return "\n".join(lines[:8]) if lines else "(schedule complete for today)"


def get_persona_client(persona: Persona) -> UnifiedPersonaClient:
    """Get or create a UnifiedPersonaClient for a persona."""
    return UnifiedPersonaClient(persona)


def get_client_stats() -> dict[str, Any]:
    """Get statistics about all persona clients."""
    return {
        "num_clients": len(_persona_clients),
        "personas": {
            name: {
                "context_tokens": _persona_usage.get(name, {}).get("context_tokens", 0),
                "context_pct": _persona_usage.get(name, {}).get("context_tokens", 0)
                / MAX_CONTEXT_TOKENS
                * 100,
                "initialized": _persona_initialized.get(name, False),
            }
            for name in _persona_clients.keys()
        },
        "compaction_threshold_pct": COMPACTION_THRESHOLD * 100,
    }


def cleanup_clients_sync():
    """Cleanup all clients (sync wrapper)."""
    if _loop is not None and _loop.is_running():
        _run_async(_cleanup_all_clients())


def set_debug_verbosity(level: int):
    """Set debug output level (0=silent, 1=summary, 2=decisions, 3=full)."""
    global DEBUG_VERBOSITY
    DEBUG_VERBOSITY = level


# ============================================================================
# #####################[SECTION 8: LEGACY COMPATIBILITY] #####################
# ============================================================================
# These functions maintain backward compatibility with run_prompt.py
# They will be deprecated once the unified system is fully integrated.


def temp_sleep(seconds=0.1):
    """Brief pause between API calls."""
    time.sleep(seconds)


def generate_prompt(curr_input, prompt_lib_file):
    """
    Legacy function for template-based prompts.
    Kept for backward compatibility during transition.
    """
    if isinstance(curr_input, str):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    with open(prompt_lib_file, "r") as f:
        prompt = f.read()

    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)

    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

    return prompt.strip()


# ============================================================================
# #####################[SECTION 9: TESTING] ##################################
# ============================================================================

if __name__ == "__main__":
    print("Unified Prompting System for Claudeville")
    print(
        f"Compaction threshold: {COMPACTION_THRESHOLD*100:.0f}% ({COMPACTION_TOKEN_LIMIT:,} tokens)"
    )
    print()

    # Simple test without full persona
    class MockScratch:
        name = "Test Persona"
        age = 25
        innate = "curious, friendly"
        learned = "A researcher interested in AI"
        lifestyle = "Works from home, enjoys coffee"
        living_area = "downtown apartment"
        currently = "testing the system"
        curr_time = None
        act_address = "test:location"
        act_description = "testing"
        f_daily_schedule = [["testing", 60], ["more testing", 120]]

    class MockPersona:
        name = "Test Persona"
        scratch = MockScratch()

    persona = MockPersona()
    initial = build_initial_prompt(persona)
    print("=== INITIAL PROMPT ===")
    print(initial[:1000])
    print("...")
    print()

    step = build_step_prompt(
        persona,
        perceptions=["A bird flies by", "Someone walks past"],
        nearby_personas=[("Alice", "reading a book"), ("Bob", "having coffee")],
        accessible_locations={
            "Home": {"living room": ["couch", "TV"], "kitchen": ["table", "fridge"]},
            "Park": {"main area": ["bench", "fountain"]},
        },
    )
    print("=== STEP PROMPT ===")
    print(step)
