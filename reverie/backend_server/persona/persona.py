"""
Persona class for Claudeville simulation.

This module defines the Persona class that powers the AI agents in the simulation.
Each persona has memory structures (spatial, associative, scratch) and uses a
unified prompting system for decision-making.

Claudeville uses one LLM call per step via UnifiedPersonaClient, replacing the
original multi-step cognitive chain.
"""

import datetime
import random

from path_finder import PathFinder
from utils import collision_block_id

import cli_interface as cli
from persona.cognitive_modules.perceive import perceive
from persona.memory_structures.associative_memory import AssociativeMemory
from persona.memory_structures.scratch import Scratch
from persona.memory_structures.spatial_memory import MemoryTree
from persona.prompt_template.claude_structure import StepResponse, UnifiedPersonaClient


class Persona:
    def __init__(self, name, folder_mem_saved=False):
        # PERSONA BASE STATE
        # <name> is the full name of the persona. This is a unique identifier for
        # the persona within Reverie.
        self.name = name

        # PERSONA MEMORY
        # If there is already memory in folder_mem_saved, we load that. Otherwise,
        # we create new memory instances.
        # <s_mem> is the persona's spatial memory.
        f_s_mem_saved = f"{folder_mem_saved}/bootstrap_memory/spatial_memory.json"
        self.s_mem = MemoryTree(f_s_mem_saved)
        # <s_mem> is the persona's associative memory.
        f_a_mem_saved = f"{folder_mem_saved}/bootstrap_memory/associative_memory"
        self.a_mem = AssociativeMemory(f_a_mem_saved)
        # <scratch> is the persona's scratch (short term memory) space.
        scratch_saved = f"{folder_mem_saved}/bootstrap_memory/scratch.json"
        self.scratch = Scratch(scratch_saved)

        # Claudeville: Unified persona client (one LLM call per step)
        self.unified_client = UnifiedPersonaClient(self)

        # Track nearby activity we've already evaluated (to avoid redundant LLM calls)
        # Format: set of (persona_name, activity_description) tuples
        self._acknowledged_nearby = set()

    def save(self, save_folder):
        """
        Save persona's current state (i.e., memory).

        INPUT:
          save_folder: The folder where we wil be saving our persona's state.
        OUTPUT:
          None
        """
        # Spatial memory contains a tree in a json format.
        # e.g., {"double studio":
        #         {"double studio":
        #           {"bedroom 2":
        #             ["painting", "easel", "closet", "bed"]}}}
        f_s_mem = f"{save_folder}/spatial_memory.json"
        self.s_mem.save(f_s_mem)

        # Associative memory contains a csv with the following rows:
        # [event.type, event.created, event.expiration, s, p, o]
        # e.g., event,2022-10-23 00:00:00,,Isabella Rodriguez,is,idle
        f_a_mem = f"{save_folder}/associative_memory"
        self.a_mem.save(f_a_mem)

        # Scratch contains non-permanent data associated with the persona. When
        # it is saved, it takes a json form. When we load it, we move the values
        # to Python variables.
        f_scratch = f"{save_folder}/scratch.json"
        self.scratch.save(f_scratch)

    def perceive(self, maze):
        """
        This function takes the current maze, and returns events that are
        happening around the persona. Importantly, perceive is guided by
        two key hyper-parameter for the  persona: 1) att_bandwidth, and
        2) retention.

        First, <att_bandwidth> determines the number of nearby events that the
        persona can perceive. Say there are 10 events that are within the vision
        radius for the persona -- perceiving all 10 might be too much. So, the
        persona perceives the closest att_bandwidth number of events in case there
        are too many events.

        Second, the persona does not want to perceive and think about the same
        event at each time step. That's where <retention> comes in -- there is
        temporal order to what the persona remembers. So if the persona's memory
        contains the current surrounding events that happened within the most
        recent retention, there is no need to perceive that again. xx

        INPUT:
          maze: Current <Maze> instance of the world.
        OUTPUT:
          a list of <ConceptNode> that are perceived and new.
            See associative_memory.py -- but to get you a sense of what it
            receives as its input: "s, p, o, desc, persona.scratch.curr_time"
        """
        return perceive(self, maze)

    async def move(self, maze, personas, curr_tile, curr_time):
        """
        Main cognitive function - decide what to do this simulation step.

        Uses UnifiedPersonaClient.step() for a single LLM call per step.
        Includes skip logic to avoid unnecessary LLM calls when:
        - Sleeping with no interruptions
        - Walking to destination
        - Continuing current action with no new nearby personas

        Returns:
            tuple: (next_tile, emoji, description, had_llm_call)
            - had_llm_call: True if LLM was called, False if action was continued
        """
        self.scratch.curr_tile = curr_tile

        # Check for new day
        new_day = False
        if not self.scratch.curr_time:
            new_day = "First day"
        elif self.scratch.curr_time.strftime("%A %B %d") != curr_time.strftime(
            "%A %B %d"
        ):
            new_day = "New day"
        self.scratch.curr_time = curr_time

        # Perceive environment (always needed for spatial memory updates)
        perceived_nodes = self.perceive(maze)
        perceptions = self._build_perception_strings(maze, perceived_nodes)
        nearby_personas = self._get_nearby_personas(maze, personas)

        # =====================================================================
        # SKIP LOGIC - Avoid unnecessary LLM calls
        # =====================================================================
        skip_result = self._should_skip_llm_call(
            new_day, perceptions, nearby_personas, maze, personas
        )
        if skip_result:
            # Log skipped personas for visibility
            from persona.prompt_template.claude_structure import DEBUG_VERBOSITY
            import cli_interface as cli

            if DEBUG_VERBOSITY >= 1:
                time_str = curr_time.strftime("%H:%M") if curr_time else ""
                emoji = skip_result[1] if len(skip_result) > 1 else "⏳"
                desc = skip_result[2] if len(skip_result) > 2 else "continuing"
                # Truncate description for display
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                print(
                    cli.c(f"  ○ {self.name}", cli.Colors.DIM)
                    + cli.c(f" {time_str} ", cli.Colors.DIM)
                    + cli.c(f"{emoji} {desc}", cli.Colors.DIM)
                )
            # Return with had_llm_call=False
            return (*skip_result, False)

        # =====================================================================
        # LLM DECISION - Need to make a new decision
        # =====================================================================
        (
            accessible_locations,
            valid_sectors,
            valid_arenas,
            valid_objects,
        ) = self._build_accessible_locations(maze)

        if new_day:
            await self._handle_new_day(new_day)

        # Build conversation context if we're in a conversation
        conversation_context = None
        if self.scratch.chatting_with and self.scratch.chat:
            # Pass existing chat lines as context so the LLM knows what was said
            conversation_context = self.scratch.chat

        # Check for nearby conversations we could join
        nearby_conversations = self._get_nearby_conversations(personas)

        step_response = await self.unified_client.step(
            perceptions=perceptions,
            nearby_personas=nearby_personas,
            accessible_locations=accessible_locations,
            valid_sectors=valid_sectors,
            valid_arenas=valid_arenas,
            valid_objects=valid_objects,
            conversation_context=conversation_context,
            nearby_conversations=nearby_conversations,
        )

        # Update acknowledged nearby after LLM call
        self._acknowledged_nearby = set(nearby_personas)

        # Process the step response (with nearby_personas for validation)
        result = self._process_step_response(
            step_response, maze, personas, nearby_personas
        )

        # Check if persona is going to sleep - trigger compaction
        if step_response.action:
            action_desc = step_response.action.description.lower()
            if "sleep" in action_desc or "go to bed" in action_desc:
                # Compact the context when going to sleep
                await self.unified_client.compact_for_sleep()

        # Return with had_llm_call=True
        return (*result, True)

    # =========================================================================
    # SKIP LOGIC
    # =========================================================================

    def _should_skip_llm_call(
        self, new_day, perceptions, nearby_personas, maze, personas
    ):
        """
        Determine if we can skip the LLM call this step.

        Priority order:
        1. New day always needs planning
        2. If walking, continue walking (unless persona nearby)
        3. If action still in progress, continue it (unless persona nearby)
        4. Nearby personas interrupt to allow social interaction
        5. Otherwise, need new decision

        Returns:
            tuple or None: If skipping, return (next_tile, emoji, description).
                          If not skipping, return None.
        """
        # Never skip on new day - need fresh planning
        if new_day:
            return None

        # Get current action info
        curr_action = self.scratch.act_description or ""
        curr_emoji = self.scratch.act_pronunciatio or "💭"
        curr_address = self.scratch.act_address or ""

        # === WALKING ===
        # If we have a planned path, continue walking (even if personas nearby -
        # we'll interact when we arrive)
        if self.scratch.act_path_set and self.scratch.planned_path:
            next_tile = self.scratch.planned_path[0]
            self.scratch.planned_path = self.scratch.planned_path[1:]
            return (next_tile, curr_emoji, f"{curr_action} @ {curr_address}")

        # === ACTION IN PROGRESS (including sleep) ===
        # If current action duration hasn't elapsed, continue it
        # Only interrupt for NEW nearby activity (not already acknowledged)
        if self._action_still_in_progress():
            if nearby_personas:
                # Check if there's any NEW activity we haven't seen yet
                current_nearby = set(nearby_personas)  # set of (name, activity) tuples
                new_activity = current_nearby - self._acknowledged_nearby

                if new_activity:
                    # New activity detected - call LLM to decide response
                    # (After LLM call, we'll update _acknowledged_nearby)
                    return None
                # All nearby activity already acknowledged - continue current action
            return self._continue_current_action(curr_emoji, curr_action, curr_address)

        # === NEARBY PERSONAS ===
        # If someone is nearby and we have no current action, consider interaction
        # (This is now only reached if action is NOT in progress)

        # Need to make a new decision
        return None

    def _action_still_in_progress(self) -> bool:
        """Check if current action duration hasn't elapsed."""
        if not self.scratch.act_start_time or not self.scratch.act_duration:
            return False

        elapsed = (
            self.scratch.curr_time - self.scratch.act_start_time
        ).total_seconds() / 60
        return elapsed < self.scratch.act_duration

    def _continue_current_action(self, emoji, description, address):
        """Return execution tuple for continuing current action without LLM call."""
        # Print skip message for visibility
        self._print_skip_message(description)
        return (self.scratch.curr_tile, emoji, f"{description} @ {address}")

    def _print_skip_message(self, description):
        """Print a dim message indicating we skipped the LLM call.

        NOTE: Only prints at verbosity level 2+ to reduce noise.
        New actions are always printed at level 1+.
        """
        from persona.prompt_template.claude_structure import DEBUG_VERBOSITY

        # Only print continuing messages at verbosity level 2+
        if DEBUG_VERBOSITY >= 2:
            color = self._get_persona_color()
            time_str = (
                self.scratch.curr_time.strftime("%H:%M")
                if self.scratch.curr_time
                else ""
            )
            short_desc = (
                description[:50] + "..." if len(description) > 50 else description
            )
            print(
                cli.c(f"  ○ {self.name}", color)
                + cli.c(f" {time_str} ", cli.Colors.DIM)
                + cli.c(f"(continuing) {short_desc}", cli.Colors.DIM)
            )

    def _get_persona_color(self) -> str:
        """Get unique color for this persona."""
        from persona.prompt_template.claude_structure import get_persona_color

        return get_persona_color(self.name)

    # =========================================================================
    # PERCEPTION HELPERS
    # =========================================================================

    def _build_perception_strings(self, maze, perceived_nodes):
        """
        Build list of perception strings from perceived ConceptNodes and environment.

        Returns a list of strings describing what the persona perceives.
        """
        perceptions = []

        # Add perceptions from ConceptNodes
        for node in perceived_nodes:
            if hasattr(node, "description") and node.description:
                # Skip self-observations
                if not node.description.startswith(self.name):
                    perceptions.append(node.description)

        # Add current location context
        tile_info = maze.access_tile(self.scratch.curr_tile)
        if tile_info.get("arena"):
            loc_str = f"You are in {tile_info.get('arena', 'unknown')}"
            if tile_info.get("sector"):
                loc_str += f" ({tile_info.get('sector')})"
            perceptions.append(loc_str)

        return perceptions

    def _get_nearby_personas(self, maze, personas):
        """
        Get list of (name, activity_key) tuples for nearby personas.

        The activity_key is (predicate, object) from the action triplet,
        which is more stable than the full description for detecting
        genuinely new activities vs just rephrased descriptions.
        """
        nearby = []
        nearby_tiles = maze.get_nearby_tiles(
            self.scratch.curr_tile, self.scratch.vision_r
        )

        for tile in nearby_tiles:
            tile_details = maze.access_tile(tile)
            if tile_details.get("events"):
                for event in tile_details["events"]:
                    # event is (subject, predicate, object, description)
                    subject = event[0]
                    # Check if this is a persona (not an object)
                    if subject in personas and subject != self.name:
                        # Use (predicate, object) as activity key - more stable than description
                        predicate = event[1] if len(event) > 1 else "is"
                        obj = event[2] if len(event) > 2 else "idle"
                        activity_key = (predicate, obj)
                        nearby.append((subject, activity_key))

        # Remove duplicates
        return list(set(nearby))

    def _get_nearby_conversations(self, personas) -> list[dict]:
        """
        Get list of nearby conversations that this persona could join.

        Checks if any nearby personas are in an active conversation
        that this persona is NOT part of.

        Returns:
            list of dicts: [{"participants": ["A", "B"], "chat": [("A", "Hi"), ...]}]
        """
        nearby_convos = []
        seen_groups = set()

        for name, _ in self._acknowledged_nearby:
            if name not in personas:
                continue

            other_persona = personas[name]
            other_scratch = other_persona.scratch

            # Check if this persona is in a conversation
            if other_scratch.chatting_with and other_scratch.chat:
                # Make sure we're not already in this conversation
                participants = other_scratch.get_conversation_participants()
                if self.name in participants:
                    continue

                # Create a unique key for this conversation group
                group_key = tuple(sorted(participants))
                if group_key in seen_groups:
                    continue
                seen_groups.add(group_key)

                nearby_convos.append(
                    {
                        "participants": participants,
                        "chat": other_scratch.chat[:10],  # Limit to last 10 lines
                        "group_id": other_scratch.conversation_group_id,
                    }
                )

        return nearby_convos

    def _build_accessible_locations(self, maze):
        """
        Build the accessible locations structure from spatial memory.

        Returns:
          accessible_locations: dict of {sector: {arena: [objects]}}
          valid_sectors: list of valid sector names
          valid_arenas: dict of {sector: [arena_names]}
          valid_objects: dict of {sector: {arena: [object_names]}}
        """
        accessible_locations = {}
        valid_sectors = []
        valid_arenas = {}
        valid_objects = {}

        # Get current world from tile
        tile_info = maze.access_tile(self.scratch.curr_tile)
        curr_world = tile_info.get("world", "")

        if not curr_world or curr_world not in self.s_mem.tree:
            return accessible_locations, valid_sectors, valid_arenas, valid_objects

        # Build from spatial memory tree
        for sector, arenas in self.s_mem.tree[curr_world].items():
            if not sector:
                continue
            valid_sectors.append(sector)
            accessible_locations[sector] = {}
            valid_arenas[sector] = []
            valid_objects[sector] = {}

            if isinstance(arenas, dict):
                for arena, objects in arenas.items():
                    if not arena:
                        continue
                    valid_arenas[sector].append(arena)
                    obj_list = objects if isinstance(objects, list) else []
                    accessible_locations[sector][arena] = obj_list
                    valid_objects[sector][arena] = obj_list

        return accessible_locations, valid_sectors, valid_arenas, valid_objects

    async def _handle_new_day(self, new_day):
        """
        Handle new day initialization - generate wake up hour and daily schedule.

        Uses LLM to generate personalized schedule based on persona's
        lifestyle, traits, and current focus.
        """
        date_str = self.scratch.curr_time.strftime("%A, %B %d, %Y")

        # Call LLM to generate personalized daily plan
        day_plan = await self.unified_client.plan_day(date_str)

        if day_plan.parse_errors:
            self._set_default_schedule()
        else:
            self.scratch.wake_up_hour = day_plan.wake_up_hour
            self.scratch.daily_req = day_plan.daily_goals
            schedule = [
                [activity, duration] for activity, duration in day_plan.schedule
            ]
            self.scratch.f_daily_schedule = schedule
            self.scratch.f_daily_schedule_hourly_org = schedule[:]

        # Add plan to memory
        thought = f"This is {self.scratch.name}'s plan for {date_str}"
        if self.scratch.daily_req:
            goals_summary = ", ".join(self.scratch.daily_req[:3])
            thought += f": {goals_summary}"

        created = self.scratch.curr_time
        expiration = self.scratch.curr_time + datetime.timedelta(days=30)
        s, p, o = (
            self.scratch.name,
            "plan",
            self.scratch.curr_time.strftime("%A %B %d"),
        )
        keywords = set(["plan", "daily", "schedule"])
        self.a_mem.add_thought(
            created, expiration, s, p, o, thought, keywords, 5, thought, None
        )

    def _set_default_schedule(self):
        """Set a default schedule when LLM planning fails."""
        default_schedule = [
            ["sleeping", 420],  # Until 7am
            ["waking up and morning routine", 60],
            ["having breakfast", 30],
            ["working on daily tasks", 180],
            ["having lunch", 60],
            ["afternoon activities", 180],
            ["relaxing", 120],
            ["having dinner", 60],
            ["evening leisure", 120],
            ["getting ready for bed", 30],
            ["sleeping", 180],
        ]
        self.scratch.f_daily_schedule = default_schedule
        self.scratch.f_daily_schedule_hourly_org = default_schedule[:]
        self.scratch.daily_req = []

    def _process_step_response(
        self, step_response: StepResponse, maze, personas, nearby_personas=None
    ):
        """
        Process the StepResponse from unified_client.step() and return execution tuple.

        This handles:
        - Updating scratch with new action details
        - Processing social decisions (conversations)
        - Resolving location names to tile coordinates
        - Storing any thoughts in memory
        - Returning the execution tuple (next_tile, pronunciatio, description)

        Args:
            nearby_personas: List of (name, activity_key) tuples of personas actually nearby.
                            Used to validate social targets.
        """
        social = step_response.social

        # Handle "continuing" flag - stay in place, keep doing what we're doing
        if step_response.continuing:
            # Store any thoughts from the response
            self._store_thoughts(step_response.thoughts)

            # Handle social even when continuing (might want to respond to someone)
            self._process_continuing_social(social, nearby_personas, personas)

            # Return current position with current action
            curr_emoji = self.scratch.act_pronunciatio or "💭"
            curr_desc = self.scratch.act_description or f"{self.name} is idle"
            curr_address = self.scratch.act_address or ""
            return (self.scratch.curr_tile, curr_emoji, f"{curr_desc} @ {curr_address}")

        # Handle parse errors - fall back to idle if we got nothing useful
        if not step_response.action:
            return self._create_idle_execution()

        action = step_response.action

        # Get current world for address construction
        tile_info = maze.access_tile(self.scratch.curr_tile)
        curr_world = tile_info.get("world", "")

        # Build the action address (world:sector:arena:object)
        act_address = (
            f"{curr_world}:{action.sector}:{action.arena}:{action.game_object}"
        )

        # Validate social target is actually nearby
        nearby_names = set()
        if nearby_personas:
            nearby_names = {name for name, _ in nearby_personas}

        # Process social decisions - build chat data if conversation is happening
        chatting_with = None
        chat = None
        chatting_with_buffer = None
        chatting_end_time = None

        # Normalize target to a list for uniform handling
        targets = []
        if social.target:
            if isinstance(social.target, list):
                targets = social.target
            else:
                targets = [social.target]

        # Check which targets are actually nearby
        nearby_targets = [
            t for t in targets if t in nearby_names or self.scratch.chatting_with == t
        ]
        missing_targets = [t for t in targets if t not in nearby_targets]

        if social.wants_to_talk and targets and social.conversation_line:
            if not nearby_targets:
                # No targets are nearby - log and skip
                cli.print_info(
                    f"  {self.name} wanted to talk to {targets} "
                    f"but none are nearby (ignoring)"
                )
            else:
                if missing_targets:
                    # Some targets missing - log but continue with those present
                    cli.print_info(
                        f"  {self.name} addressing {nearby_targets} "
                        f"(missing: {missing_targets})"
                    )

                # Starting or continuing a conversation
                # For chatting_with, use first target (primary addressee)
                chatting_with = nearby_targets[0]

                # Build chat list - either append to existing or start new
                if self.scratch.chat:
                    chat = self.scratch.chat.copy()
                else:
                    chat = []

                # Add our line to the conversation
                chat.append([self.name, social.conversation_line])

                # Set chatting buffer for ALL nearby targets (for vision tracking)
                chatting_with_buffer = {
                    t: self.scratch.vision_r for t in nearby_targets
                }

                # Set conversation end time based on action duration
                chatting_end_time = self.scratch.curr_time + datetime.timedelta(
                    minutes=action.duration_minutes
                )

                # Print conversation to CLI
                cli.print_conversation_line(self.name, social.conversation_line)

        elif social.conversation_line and not social.wants_to_talk:
            # Just saying something (no formal conversation)
            if self.scratch.chat:
                chat = self.scratch.chat.copy()
            else:
                chat = []
            chat.append([self.name, social.conversation_line])
            cli.print_conversation_line(self.name, social.conversation_line)

        # Update scratch with the new action
        self.scratch.add_new_action(
            action_address=act_address,
            action_duration=action.duration_minutes,
            action_description=action.description,
            action_pronunciatio=action.emoji,
            action_event=action.event,
            chatting_with=chatting_with,
            chat=chat,
            chatting_with_buffer=chatting_with_buffer,
            chatting_end_time=chatting_end_time,
            act_obj_description=None,
            act_obj_pronunciatio=None,
            act_obj_event=(None, None, None),
        )

        # Store any thoughts from the response
        self._store_thoughts(step_response.thoughts)

        # Resolve location to tile coordinates (adapted from execute.py)
        next_tile = self._resolve_location_to_tile(act_address, maze, personas)

        # Build description string
        description = f"{action.description} @ {act_address}"

        return (next_tile, action.emoji, description)

    def _resolve_location_to_tile(self, act_address, maze, personas):
        """
        Resolve an action address to tile coordinates.

        Adapted from execute.py logic.
        """
        # If path is already set and valid, use the planned path
        if self.scratch.act_path_set and self.scratch.planned_path:
            ret = self.scratch.planned_path[0]
            self.scratch.planned_path = self.scratch.planned_path[1:]
            return ret

        # Check if we're already at the target location - stay in place
        # This prevents unnecessary movement when doing the same activity
        curr_tile_info = maze.access_tile(self.scratch.curr_tile)
        curr_address_parts = [
            curr_tile_info.get("world", ""),
            curr_tile_info.get("sector", ""),
            curr_tile_info.get("arena", ""),
        ]
        curr_arena_address = ":".join(curr_address_parts)

        # Extract arena-level address from target (first 3 parts)
        target_parts = act_address.split(":")
        target_arena_address = (
            ":".join(target_parts[:3]) if len(target_parts) >= 3 else act_address
        )

        if curr_arena_address == target_arena_address:
            # Already at the target arena - stay in place
            return self.scratch.curr_tile

        # Need to find a path to the target location
        path_finder = PathFinder(maze.collision_maze, collision_block_id)

        # Check for special address types
        if "<persona>" in act_address:
            # Moving to interact with another persona
            target_name = act_address.split("<persona>")[-1].strip()
            if target_name in personas:
                target_tile = personas[target_name].scratch.curr_tile
                path = path_finder.find_path(self.scratch.curr_tile, target_tile)
                if len(path) > 1:
                    self.scratch.planned_path = path[1:]
                    self.scratch.act_path_set = True
                    return path[1] if len(path) > 1 else path[0]
            return self.scratch.curr_tile

        if "<waiting>" in act_address:
            # Waiting in place
            return self.scratch.curr_tile

        if "<random>" in act_address:
            # Random location within area
            clean_address = ":".join(act_address.split(":")[:-1])
            act_address = clean_address

        # Standard location resolution
        target_tiles = None
        if act_address in maze.address_tiles:
            target_tiles = list(maze.address_tiles[act_address])
        else:
            # Try partial address matching (without object)
            parts = act_address.split(":")
            for i in range(len(parts), 0, -1):
                partial = ":".join(parts[:i])
                if partial in maze.address_tiles:
                    target_tiles = list(maze.address_tiles[partial])
                    break

        if not target_tiles:
            # Fallback: stay in place
            return self.scratch.curr_tile

        # Sample a few target tiles and pick the closest unoccupied one
        if len(target_tiles) > 4:
            target_tiles = random.sample(target_tiles, 4)

        # Avoid tiles occupied by other personas
        persona_names = set(personas.keys())
        unoccupied_tiles = []
        for tile in target_tiles:
            tile_events = maze.access_tile(tile).get("events", [])
            occupied = any(event[0] in persona_names for event in tile_events)
            if not occupied:
                unoccupied_tiles.append(tile)

        if unoccupied_tiles:
            target_tiles = unoccupied_tiles

        # Find path to nearest target tile
        path, closest_tile = path_finder.find_path_to_nearest(
            self.scratch.curr_tile, target_tiles
        )

        if path and len(path) > 1:
            self.scratch.planned_path = path[1:]
            self.scratch.act_path_set = True
            return path[1]

        return self.scratch.curr_tile

    def _store_thoughts(self, thoughts):
        """
        Store thoughts from StepResponse into associative memory.
        """
        for thought in thoughts:
            if not thought.content:
                continue

            created = self.scratch.curr_time
            expiration = self.scratch.curr_time + datetime.timedelta(days=30)
            s, p, o = self.scratch.name, "thought", thought.content[:50]
            keywords = set(["thought", "reflection"])

            self.a_mem.add_thought(
                created,
                expiration,
                s,
                p,
                o,
                thought.content,
                keywords,
                thought.importance,
                thought.content,
                None,
            )

    def _process_continuing_social(self, social, nearby_personas, personas):
        """
        Process social decisions when persona is continuing their current activity.

        This allows the persona to respond in conversation even while staying in place.
        """
        if not social or not social.conversation_line:
            return

        # Validate social target is actually nearby
        nearby_names = set()
        if nearby_personas:
            nearby_names = {name for name, _ in nearby_personas}

        # Normalize target to a list for uniform handling
        targets = []
        if social.target:
            if isinstance(social.target, list):
                targets = social.target
            else:
                targets = [social.target]

        # Check which targets are actually nearby
        nearby_targets = [
            t for t in targets if t in nearby_names or self.scratch.chatting_with == t
        ]

        if social.wants_to_talk and targets and social.conversation_line:
            if not nearby_targets:
                cli.print_info(
                    f"  {self.name} wanted to talk to {targets} "
                    f"but none are nearby (ignoring)"
                )
            else:
                # Add line to existing conversation
                if self.scratch.chat:
                    self.scratch.chat.append([self.name, social.conversation_line])
                else:
                    self.scratch.chat = [[self.name, social.conversation_line]]

                # Update chatting_with if starting new conversation
                if not self.scratch.chatting_with:
                    self.scratch.chatting_with = nearby_targets[0]
                    self.scratch.chatting_with_buffer = {
                        t: self.scratch.vision_r for t in nearby_targets
                    }

                cli.print_conversation_line(self.name, social.conversation_line)

        elif social.conversation_line and not social.wants_to_talk:
            # Just saying something (no formal conversation)
            if self.scratch.chat:
                self.scratch.chat.append([self.name, social.conversation_line])
            else:
                self.scratch.chat = [[self.name, social.conversation_line]]
            cli.print_conversation_line(self.name, social.conversation_line)

    def _create_idle_execution(self):
        """
        Create a default idle execution when we can't get a proper response.
        """
        return (self.scratch.curr_tile, "💭", f"{self.name} is idle")
