"""
Original Author: Joon Sung Park (joonspk@stanford.edu)
Heavily modified for Claudeville (Claude CLI port)

File: reverie.py
Description: Main program for running generative agent simulations.
"""

import asyncio
import datetime
import json
import math
import os
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

from flask import Flask, jsonify, request

import cli_interface as cli
from maze import Maze
from persona.prompt_template.claude_structure import _run_async
from utils import (
    debug,
    fs_storage,
    fs_storage_base,
    fs_storage_runs,
    fs_temp_storage,
)

from persona.persona import Persona


##############################################################################
#                           CONVERSATION GROUPS                              #
##############################################################################


@dataclass
class ConversationGroup:
    """
    Represents an active multi-party conversation.

    Managed centrally by ReverieServer to enable 3+ persona conversations.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    participants: set = field(default_factory=set)
    chat: list = field(default_factory=list)  # [(speaker, line), ...]
    location_tile: tuple = (0, 0)  # Anchor location (first participant's tile)
    started_at: datetime.datetime = None
    end_time: datetime.datetime = None
    last_activity: datetime.datetime = None  # Last time a new message was added
    stale_steps: int = 0  # Steps since last new message

    def add_participant(self, name: str) -> bool:
        """Add a participant to the conversation. Returns True if newly added."""
        if name in self.participants:
            return False
        self.participants.add(name)
        return True

    def add_line(
        self, speaker: str, line: str, curr_time: datetime.datetime = None
    ) -> bool:
        """Add a chat line. Returns True if added (not duplicate)."""
        entry = (speaker, line)
        if entry not in [(spk, txt) for spk, txt in self.chat]:
            self.chat.append(entry)
            self.last_activity = curr_time
            self.stale_steps = 0  # Reset stale counter on new activity
            return True
        return False

    def merge_lines(self, lines: list, curr_time: datetime.datetime = None) -> int:
        """Merge lines from another source. Returns count of new lines added."""
        added = 0
        for speaker, line in lines:
            if self.add_line(speaker, line, curr_time):
                added += 1
        return added

    def get_participants_str(self) -> str:
        """Get participant names as comma-separated string."""
        return ", ".join(sorted(self.participants))

    def remove_participant(self, name: str) -> bool:
        """Remove a participant from the conversation. Returns True if removed."""
        if name in self.participants:
            self.participants.discard(name)
            return True
        return False


def _tile_distance(tile1: tuple, tile2: tuple) -> float:
    """Calculate Euclidean distance between two tiles."""
    if not tile1 or not tile2:
        return float("inf")
    return math.sqrt((tile1[0] - tile2[0]) ** 2 + (tile1[1] - tile2[1]) ** 2)


def _are_within_range(tile1: tuple, tile2: tuple, vision_r: int = 4) -> bool:
    """Check if two tiles are within vision range of each other."""
    return _tile_distance(tile1, tile2) <= vision_r


# Backend HTTP server port
BACKEND_PORT = 5000

##############################################################################
#                                  REVERIE                                   #
##############################################################################


class ReverieServer:
    def __init__(self, fork_sim_code, sim_code):
        # FORKING FROM A PRIOR SIMULATION:
        # <fork_sim_code> indicates the simulation we are forking from.
        # Base templates are in storage/base/, simulation runs go to storage/runs/
        self.fork_sim_code = fork_sim_code

        # Check if fork is a base template or a previous run
        if os.path.exists(f"{fs_storage_base}/{self.fork_sim_code}"):
            fork_folder = f"{fs_storage_base}/{self.fork_sim_code}"
        else:
            fork_folder = f"{fs_storage_runs}/{self.fork_sim_code}"

        # <sim_code> indicates our current simulation. Runs always go to storage/runs/
        self.sim_code = sim_code
        sim_folder = f"{fs_storage_runs}/{self.sim_code}"
        if not os.path.exists(sim_folder):
            shutil.copytree(fork_folder, sim_folder)

        # Create movement folder for this run (not in base template)
        os.makedirs(f"{sim_folder}/movement", exist_ok=True)

        with open(f"{sim_folder}/reverie/meta.json") as json_file:
            reverie_meta = json.load(json_file)

        with open(f"{sim_folder}/reverie/meta.json", "w") as outfile:
            reverie_meta["fork_sim_code"] = fork_sim_code
            outfile.write(json.dumps(reverie_meta, indent=2))

        # LOADING REVERIE'S GLOBAL VARIABLES
        # The start datetime of the Reverie:
        # <start_datetime> is the datetime instance for the start datetime of
        # the Reverie instance. Once it is set, this is not really meant to
        # change. It takes a string date in the following example form:
        # "June 25, 2022"
        # e.g., ...strptime(June 25, 2022, "%B %d, %Y")
        self.start_time = datetime.datetime.strptime(
            f"{reverie_meta['start_date']}, 00:00:00", "%B %d, %Y, %H:%M:%S"
        )
        # <curr_time> is the datetime instance that indicates the game's current
        # time. This gets incremented by <sec_per_step> amount everytime the world
        # progresses (that is, everytime curr_env_file is recieved).
        self.curr_time = datetime.datetime.strptime(
            reverie_meta["curr_time"], "%B %d, %Y, %H:%M:%S"
        )
        # <sec_per_step> denotes the number of seconds in game time that each
        # step moves foward.
        self.sec_per_step = reverie_meta["sec_per_step"]

        # <maze> is the main Maze instance. Note that we pass in the maze_name
        # (e.g., "double_studio") to instantiate Maze.
        # e.g., Maze("double_studio")
        self.maze = Maze(reverie_meta["maze_name"])

        # <step> denotes the number of steps that our game has taken. A step here
        # literally translates to the number of moves our personas made in terms
        # of the number of tiles.
        self.step = reverie_meta["step"]

        # SETTING UP PERSONAS IN REVERIE
        # <personas> is a dictionary that takes the persona's full name as its
        # keys, and the actual persona instance as its values.
        # This dictionary is meant to keep track of all personas who are part of
        # the Reverie instance.
        # e.g., ["Isabella Rodriguez"] = Persona("Isabella Rodriguezs")
        self.personas = dict()
        # <personas_tile> is a dictionary that contains the tile location of
        # the personas (!-> NOT px tile, but the actual tile coordinate).
        # The tile take the form of a set, (row, col).
        # e.g., ["Isabella Rodriguez"] = (58, 39)
        self.personas_tile = dict()

        # # <persona_convo_match> is a dictionary that describes which of the two
        # # personas are talking to each other. It takes a key of a persona's full
        # # name, and value of another persona's full name who is talking to the
        # # original persona.
        # # e.g., dict["Isabella Rodriguez"] = ["Maria Lopez"]
        # self.persona_convo_match = dict()
        # # <persona_convo> contains the actual content of the conversations. It
        # # takes as keys, a pair of persona names, and val of a string convo.
        # # Note that the key pairs are *ordered alphabetically*.
        # # e.g., dict[("Adam Abraham", "Zane Xu")] = "Adam: baba \n Zane:..."
        # self.persona_convo = dict()

        # Loading in all personas.
        # Try to get positions from meta (new way) or fall back to environment file (old way)
        persona_tiles = reverie_meta.get("persona_tiles")
        if not persona_tiles:
            # Fallback: load from environment file (for old simulations)
            init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
            if os.path.exists(init_env_file):
                init_env = json.load(open(init_env_file))
                persona_tiles = {
                    name: [init_env[name]["x"], init_env[name]["y"]]
                    for name in reverie_meta["persona_names"]
                }
            else:
                # Last resort: find most recent environment file
                env_dir = f"{sim_folder}/environment"
                env_files = [f for f in os.listdir(env_dir) if f.endswith(".json")]
                if env_files:
                    latest = max(env_files, key=lambda f: int(f.replace(".json", "")))
                    init_env = json.load(open(f"{env_dir}/{latest}"))
                    persona_tiles = {
                        name: [init_env[name]["x"], init_env[name]["y"]]
                        for name in reverie_meta["persona_names"]
                    }
                else:
                    raise FileNotFoundError(
                        f"No persona position data found for {self.sim_code}"
                    )

        for persona_name in reverie_meta["persona_names"]:
            persona_folder = f"{sim_folder}/personas/{persona_name}"
            p_x, p_y = persona_tiles[persona_name]
            curr_persona = Persona(persona_name, persona_folder)

            self.personas[persona_name] = curr_persona
            self.personas_tile[persona_name] = (p_x, p_y)
            self.maze.tiles[p_y][p_x]["events"].add(
                curr_persona.scratch.get_curr_event_and_desc()
            )

        # REVERIE SETTINGS PARAMETERS:
        # <server_sleep> denotes the amount of time that our while loop rests each
        # cycle; this is to not kill our machine.
        self.server_sleep = 0.1

        # SIGNALING THE FRONTEND SERVER:
        # curr_sim_code.json contains the current simulation code, and
        # curr_step.json contains the current step of the simulation. These are
        # used to communicate the code and step information to the frontend.
        # Note that step file is removed as soon as the frontend opens up the
        # simulation.
        curr_sim_code = dict()
        curr_sim_code["sim_code"] = self.sim_code
        with open(f"{fs_temp_storage}/curr_sim_code.json", "w") as outfile:
            outfile.write(json.dumps(curr_sim_code, indent=2))

        curr_step = dict()
        curr_step["step"] = self.step
        with open(f"{fs_temp_storage}/curr_step.json", "w") as outfile:
            outfile.write(json.dumps(curr_step, indent=2))

        # HTTP SERVER SETUP
        # Flask app for handling step requests from frontend
        self.flask_app = Flask(__name__)
        self.flask_app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
        self._setup_flask_routes()

        # Track game object cleanup between steps
        self._game_obj_cleanup = dict()

        # Lock to prevent concurrent step processing (CLI vs HTTP)
        self._step_lock = threading.Lock()

        # Queue of pending movements for frontend to display
        # Each entry is a movements dict from _process_step
        self._pending_movements = []
        self._movements_lock = threading.Lock()

        # Active conversation groups for multi-party conversations
        # Maps group_id -> ConversationGroup
        self.active_conversations: dict[str, ConversationGroup] = {}

    def _setup_flask_routes(self):
        """Set up Flask HTTP endpoints for frontend communication."""

        @self.flask_app.route("/movements", methods=["GET"])
        def handle_movements():
            """
            Return pending movements for frontend to animate.
            Frontend polls this endpoint to get movement data.
            Backend (CLI) drives the simulation and queues movements here.
            """
            with self._movements_lock:
                if self._pending_movements:
                    # Return oldest pending movement
                    movement = self._pending_movements.pop(0)
                    return jsonify(movement)
                else:
                    # No pending movements
                    return jsonify({"empty": True, "step": self.step})

        @self.flask_app.route("/status", methods=["GET"])
        def handle_status():
            """Return current simulation status."""
            return jsonify(
                {
                    "sim_code": self.sim_code,
                    "step": self.step,
                    "curr_time": self.curr_time.strftime("%B %d, %Y, %H:%M:%S"),
                    "personas": list(self.personas.keys()),
                }
            )

        @self.flask_app.route("/save", methods=["POST"])
        def handle_save():
            """Save simulation state."""
            self.save()
            return jsonify({"status": "saved", "step": self.step})

        @self.flask_app.route("/simulate", methods=["POST"])
        def handle_simulate():
            """
            Run simulation steps on demand from frontend.
            Request body: {"steps": N} where N is number of steps to simulate.
            Returns immediately after queueing the steps.
            """
            # Check if a step is already in progress (non-blocking)
            if not self._step_lock.acquire(blocking=False):
                # Step already running - return current state without running more
                return jsonify(
                    {
                        "status": "busy",
                        "message": "Step already in progress",
                        "current_step": self.step,
                        "queued_movements": len(self._pending_movements),
                    }
                )

            try:
                data = request.get_json() or {}
                num_steps = min(data.get("steps", 1), 10)  # Cap at 10 steps per request

                # Build environment data from current state
                environment = {}
                for persona_name in self.personas:
                    tile = self.personas_tile[persona_name]
                    environment[persona_name] = {"x": tile[0], "y": tile[1]}

                # Run the steps (movements get queued automatically)
                for i in range(num_steps):
                    # Print step header like CLI does
                    cli.print_step_start(self.step, self.curr_time)
                    self._process_step_unlocked({"environment": environment})
                    # Update environment for next step
                    for persona_name in self.personas:
                        tile = self.personas_tile[persona_name]
                        environment[persona_name] = {"x": tile[0], "y": tile[1]}

                return jsonify(
                    {
                        "status": "ok",
                        "steps_run": num_steps,
                        "current_step": self.step,
                        "queued_movements": len(self._pending_movements),
                    }
                )
            finally:
                self._step_lock.release()

        @self.flask_app.route("/saves", methods=["GET"])
        def handle_list_saves():
            """List available save files."""
            saves = []
            runs_dir = fs_storage
            if os.path.exists(runs_dir):
                for sim_name in os.listdir(runs_dir):
                    meta_path = f"{runs_dir}/{sim_name}/reverie/meta.json"
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path) as f:
                                meta = json.load(f)
                            saves.append(
                                {
                                    "sim_code": sim_name,
                                    "step": meta.get("step", 0),
                                    "curr_time": meta.get("curr_time", ""),
                                    "personas": meta.get("persona_names", []),
                                }
                            )
                        except (OSError, json.JSONDecodeError, KeyError):
                            pass
            return jsonify({"saves": saves})

    def _process_step(self, data):
        """
        Process one simulation step. This is the core logic extracted from
        start_server() but designed for HTTP request/response instead of
        file-based polling.

        Args:
            data: dict with 'step', 'sim_code', 'environment' keys
                  environment contains persona positions: {name: {x, y, maze}}

        Returns:
            dict with 'persona' movements and 'meta' information

        Thread-safe: Uses _step_lock to prevent concurrent processing.
        """
        with self._step_lock:
            return self._process_step_unlocked(data)

    def _process_step_unlocked(self, data):
        """Internal step processing (must hold _step_lock)."""
        # Note: We ignore data["step"] - the backend is authoritative.
        # This allows CLI "run" commands to work alongside frontend requests.
        environment = data.get("environment", {})

        # Clean up game object events from previous cycle
        for key, val in self._game_obj_cleanup.items():
            self.maze.turn_event_from_tile_idle(key, val)
        self._game_obj_cleanup = dict()

        # Update persona positions in backend to match frontend
        for persona_name, persona in self.personas.items():
            curr_tile = self.personas_tile[persona_name]
            new_tile = (
                environment[persona_name]["x"],
                environment[persona_name]["y"],
            )

            # Move persona on backend tile map
            self.personas_tile[persona_name] = new_tile
            self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
            self.maze.add_event_from_tile(
                persona.scratch.get_curr_event_and_desc(), new_tile
            )

            # If persona reached destination, activate object action
            if not persona.scratch.planned_path:
                self._game_obj_cleanup[
                    persona.scratch.get_curr_obj_event_and_desc()
                ] = new_tile
                self.maze.add_event_from_tile(
                    persona.scratch.get_curr_obj_event_and_desc(), new_tile
                )
                blank = (
                    persona.scratch.get_curr_obj_event_and_desc()[0],
                    None,
                    None,
                    None,
                )
                self.maze.remove_event_from_tile(blank, new_tile)

        # Run cognitive pipeline for all personas in parallel
        movements = {"persona": {}, "meta": {}}

        async def run_persona_move(name, persona):
            """Run a single persona's move asynchronously."""
            (
                next_tile,
                pronunciatio,
                description,
                had_llm_call,
            ) = await persona.move(
                self.maze,
                self.personas,
                self.personas_tile[name],
                self.curr_time,
            )
            return (
                name,
                next_tile,
                pronunciatio,
                description,
                persona.scratch.chat,
                had_llm_call,
            )

        async def run_all_personas():
            """Run all personas in parallel using asyncio.gather."""
            tasks = [
                run_persona_move(name, persona)
                for name, persona in self.personas.items()
            ]
            return await asyncio.gather(*tasks)

        # Run all persona moves in parallel using the shared event loop
        # This ensures Flask thread uses the same loop as Claude SDK clients
        results = _run_async(run_all_personas())

        # Track if any persona had an LLM call (new action)
        any_llm_call = False
        active_personas = []  # Track which personas made LLM decisions
        for name, next_tile, pronunciatio, description, chat, had_llm_call in results:
            movements["persona"][name] = {
                "movement": next_tile,
                "pronunciatio": pronunciatio,
                "description": description,
                "chat": chat,
                "had_action": had_llm_call,  # Mark individual persona's LLM status
            }
            # Update backend position state with new tile
            self.personas_tile[name] = next_tile
            if had_llm_call:
                any_llm_call = True
                active_personas.append(name)

        # CONVERSATION SYNCHRONIZATION
        # After all personas have moved in parallel, synchronize their chats.
        # If Klaus is chatting with Maria and Maria is chatting with Klaus,
        # merge their chat lists so both have the full conversation.
        try:
            self._synchronize_conversations()
        except Exception as e:
            cli.print_error(f"Error in conversation sync: {e}")
            import traceback

            traceback.print_exc()

        # Update movements with synchronized chat data
        # This ensures frontend receives the complete merged conversation
        for name, persona in self.personas.items():
            if name in movements["persona"] and persona.scratch.chat:
                movements["persona"][name]["chat"] = persona.scratch.chat

        # Add meta information (step is sent BEFORE increment so frontend knows what step this was)
        movements["meta"]["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
        movements["meta"]["step"] = self.step  # Current step being processed
        movements["meta"][
            "had_new_action"
        ] = any_llm_call  # True if any persona made new decision
        movements["meta"][
            "active_personas"
        ] = active_personas  # List of personas who made decisions

        # Add active conversation groups for frontend display
        movements["meta"]["conversations"] = {
            group_id: {
                "participants": list(group.participants),
                "line_count": len(group.chat),
            }
            for group_id, group in self.active_conversations.items()
        }

        # Advance simulation state
        self.step += 1
        self.curr_time += datetime.timedelta(seconds=self.sec_per_step)

        # Update temp storage for recovery (can be removed later)
        with open(f"{fs_temp_storage}/curr_step.json", "w") as outfile:
            outfile.write(json.dumps({"step": self.step}, indent=2))

        # Queue movements for frontend to poll
        with self._movements_lock:
            self._pending_movements.append(movements)

        return movements

    def _synchronize_conversations(self):
        """
        Synchronize chat histories between personas who are in conversation.

        After all personas have moved in parallel, this method:
        1. Removes participants who have moved out of range
        2. Manages ConversationGroup objects for multi-party conversations
        3. Detects conversations and groups participants together (with proximity checks)
        4. Merges chat lines so all participants see the full conversation
        5. Stores completed conversations in all participants' memories

        This is critical because personas run in parallel and don't see each
        other's dialogue responses within a single simulation step.
        """
        # Step 0: Remove out-of-range participants from existing groups
        # Use a larger range for CONTINUING conversations than for starting new ones
        # This prevents conversations from breaking when people briefly move around
        CONVERSATION_CONTINUATION_RANGE = 8  # More lenient than vision_r (typically 4)

        for group_id, group in list(self.active_conversations.items()):
            if len(group.participants) < 2:
                continue

            # Find participants who are no longer within range of ANY other participant
            to_remove = []
            for participant in group.participants:
                if participant not in self.personas:
                    to_remove.append(participant)
                    continue

                my_tile = self.personas_tile.get(participant)

                # Check if this participant is within extended range of at least one other
                has_nearby_partner = False
                for other in group.participants:
                    if other == participant or other not in self.personas:
                        continue
                    other_tile = self.personas_tile.get(other)
                    if _are_within_range(
                        my_tile, other_tile, CONVERSATION_CONTINUATION_RANGE
                    ):
                        has_nearby_partner = True
                        break

                if not has_nearby_partner:
                    to_remove.append(participant)

            # Remove out-of-range participants and end their conversation
            for participant in to_remove:
                if participant in self.personas:
                    persona = self.personas[participant]
                    cli.print_info(
                        f"  {participant} left conversation [{group_id}] (moved out of range)"
                    )
                    self._end_and_store_conversation(persona)
                    persona.scratch.conversation_group_id = None
                group.remove_participant(participant)

        # Step 1: Build mapping of persona -> existing group from scratch
        # This ensures we reuse existing groups instead of creating new ones
        persona_to_group: dict[str, str] = {}
        for name, persona in self.personas.items():
            group_id = persona.scratch.conversation_group_id
            if group_id and group_id in self.active_conversations:
                persona_to_group[name] = group_id

        # Step 2: Collect all active conversationalists and their chat lines
        active_chatters = {}  # {name: (partner, chat_lines)}
        for name, persona in self.personas.items():
            if persona.scratch.chatting_with and persona.scratch.chat:
                active_chatters[name] = (
                    persona.scratch.chatting_with,
                    persona.scratch.chat or [],
                )

        # Step 3: Find or create conversation groups (with proximity validation)
        for name, (partner, chat_lines) in active_chatters.items():
            group = None
            group_id = None
            my_tile = self.personas_tile.get(name)
            vision_r = self.personas[name].scratch.vision_r

            # Check if this persona is already in a group
            if name in persona_to_group:
                group_id = persona_to_group[name]
                group = self.active_conversations.get(group_id)

            # Check if partner is already in a group we should join
            # BUT only if we're within range of them!
            if not group and partner in persona_to_group:
                partner_tile = self.personas_tile.get(partner)
                if _are_within_range(my_tile, partner_tile, vision_r):
                    group_id = persona_to_group[partner]
                    group = self.active_conversations.get(group_id)
                    if group:
                        group.add_participant(name)
                        persona_to_group[name] = group_id
                        # Update persona's group reference
                        self.personas[name].scratch.conversation_group_id = group_id
                else:
                    # Partner too far away - can't join their conversation
                    # End this persona's conversation attempt
                    cli.print_info(
                        f"  {name} tried to chat with {partner} but they're too far away"
                    )
                    self._end_and_store_conversation(self.personas[name])
                    continue

            # Create new group if neither is in one
            if not group:
                # Validate partner is within range before creating group
                partner_tile = self.personas_tile.get(partner)
                if partner in self.personas and not _are_within_range(
                    my_tile, partner_tile, vision_r
                ):
                    # Can't start conversation - partner too far
                    cli.print_info(
                        f"  {name} tried to chat with {partner} but they're too far away"
                    )
                    self._end_and_store_conversation(self.personas[name])
                    continue

                group = ConversationGroup(
                    participants={name},
                    location_tile=self.personas_tile.get(name, (0, 0)),
                    started_at=self.curr_time,
                )
                self.active_conversations[group.id] = group
                persona_to_group[name] = group.id
                group_id = group.id
                # Update persona's group reference
                self.personas[name].scratch.conversation_group_id = group_id

            # Add all conversation targets to group (supports multi-target/broadcast)
            # chatting_with_buffer contains all targets from persona's social decision
            all_targets = set()
            if partner:
                all_targets.add(partner)
            # Also add anyone from the chatting_with_buffer (for multi-target)
            if self.personas[name].scratch.chatting_with_buffer:
                all_targets.update(
                    self.personas[name].scratch.chatting_with_buffer.keys()
                )

            for target in all_targets:
                if target in self.personas and target not in group.participants:
                    target_tile = self.personas_tile.get(target)
                    if _are_within_range(my_tile, target_tile, vision_r):
                        group.add_participant(target)
                        persona_to_group[target] = group_id
                        # Update target's group reference
                        self.personas[target].scratch.conversation_group_id = group_id

            # Merge this persona's chat lines into the group
            group.merge_lines(chat_lines, self.curr_time)

        # Step 4: Synchronize all group members' chat lists
        for group_id, group in list(self.active_conversations.items()):
            if len(group.participants) < 2:
                continue

            # Get the full merged chat from the group
            full_chat = group.chat

            # Update each participant's scratch with the full conversation
            for participant_name in list(group.participants):
                if participant_name not in self.personas:
                    continue

                persona = self.personas[participant_name]
                persona.scratch.merge_chat_lines(full_chat)
                # Ensure group reference is set
                persona.scratch.conversation_group_id = group_id

                # If they weren't in a conversation, set them up
                if not persona.scratch.chatting_with:
                    # Pick the nearest other participant as their "chatting_with"
                    my_tile = self.personas_tile.get(participant_name)
                    others = [p for p in group.participants if p != participant_name]
                    # Sort by distance to pick nearest
                    others.sort(
                        key=lambda p: _tile_distance(my_tile, self.personas_tile.get(p))
                    )
                    if others:
                        persona.scratch.chatting_with = others[0]
                        persona.scratch.chat = list(full_chat)
                        persona.scratch.chatting_with_buffer = {
                            p: persona.scratch.vision_r for p in others
                        }
                        if group.end_time:
                            persona.scratch.chatting_end_time = group.end_time
                        else:
                            persona.scratch.chatting_end_time = (
                                self.curr_time + datetime.timedelta(minutes=5)
                            )

            # Debug output
            cli.print_info(
                f"  Conversation group [{group.id}]: "
                f"{group.get_participants_str()} ({len(group.chat)} lines)"
            )

        # Step 5: Handle one-sided conversations (A talks to B, B hasn't responded)
        # Only initiate if both are within range
        for name, (partner, chat_lines) in active_chatters.items():
            if partner not in self.personas:
                continue

            persona_b = self.personas[partner]

            # If B isn't chatting yet, set them up to respond (if within range)
            if not persona_b.scratch.chatting_with and chat_lines:
                my_tile = self.personas_tile.get(name)
                partner_tile = self.personas_tile.get(partner)
                vision_r = self.personas[name].scratch.vision_r

                if not _are_within_range(my_tile, partner_tile, vision_r):
                    # Too far apart - don't initiate conversation
                    continue

                persona_b.scratch.chatting_with = name
                persona_b.scratch.chat = []
                persona_b.scratch.merge_chat_lines(chat_lines)
                persona_b.scratch.chatting_with_buffer = {
                    name: persona_b.scratch.vision_r
                }

                persona_a = self.personas[name]
                if persona_a.scratch.chatting_end_time:
                    persona_b.scratch.chatting_end_time = (
                        persona_a.scratch.chatting_end_time
                    )
                else:
                    persona_b.scratch.chatting_end_time = (
                        self.curr_time + datetime.timedelta(minutes=5)
                    )

                cli.print_info(
                    f"  Initiated conversation: {name} -> {partner} "
                    f"({len(chat_lines)} lines shared)"
                )

        # Step 6: Check for conversations that have ended (by time)
        # Each person ends individually - others can still respond before they also end
        # This allows farewell exchanges where both parties can say goodbye
        ended_groups = []
        for name, persona in self.personas.items():
            scratch = persona.scratch
            if scratch.chatting_end_time and scratch.curr_time:
                if scratch.curr_time >= scratch.chatting_end_time:
                    group_id = scratch.conversation_group_id
                    group = (
                        self.active_conversations.get(group_id) if group_id else None
                    )

                    # End just for this persona - remove them from the group
                    # Others can still respond; group cleaned up when empty
                    self._end_and_store_conversation(persona)
                    scratch.conversation_group_id = None

                    if group:
                        group.remove_participant(name)
                        # If only one person left, they can still send one more message
                        # Group will be cleaned up in Step 7 when it becomes empty/stale

        # Step 7: Clean up orphaned, single-participant, or stale groups
        STALE_THRESHOLD = 5  # Auto-end after 5 steps of no new messages
        for group_id, group in list(self.active_conversations.items()):
            if group_id in ended_groups:
                continue  # Already marked for deletion

            # Increment stale counter for groups with no new activity
            group.stale_steps += 1

            # Remove individual participants who have stopped chatting
            # (their chatting_with is now None, meaning they started a different action)
            inactive_participants = []
            for participant in list(group.participants):
                if participant in self.personas:
                    scratch = self.personas[participant].scratch
                    if not scratch.chatting_with:
                        inactive_participants.append(participant)

            for participant in inactive_participants:
                self._end_and_store_conversation(self.personas[participant])
                self.personas[participant].scratch.conversation_group_id = None
                group.remove_participant(participant)

            # Clean up single-participant groups (everyone else left)
            if len(group.participants) < 2:
                ended_groups.append(group_id)
                for participant in list(group.participants):
                    if participant in self.personas:
                        self._end_and_store_conversation(self.personas[participant])
                        self.personas[participant].scratch.conversation_group_id = None
                continue

            # Auto-end stale conversations (no new messages for too long)
            if group.stale_steps >= STALE_THRESHOLD:
                cli.print_info(
                    f"  Auto-ending stale conversation [{group_id}] "
                    f"({group.stale_steps} steps inactive)"
                )
                ended_groups.append(group_id)
                for participant in list(group.participants):
                    if participant in self.personas:
                        self._end_and_store_conversation(self.personas[participant])
                        self.personas[participant].scratch.conversation_group_id = None
                continue

        # Clean up ended conversation groups
        for group_id in ended_groups:
            if group_id in self.active_conversations:
                del self.active_conversations[group_id]

    def _end_and_store_conversation(self, persona):
        """
        End a conversation and store it in the persona's associative memory.

        Called when chatting_end_time has been reached.
        """
        partner, chat_lines = persona.scratch.end_conversation()
        if not partner or not chat_lines:
            return

        # Build a description of the conversation
        num_lines = len(chat_lines)
        description = f"Conversation with {partner} ({num_lines} exchanges)"

        # Create keywords from the conversation
        keywords = set([partner.lower(), "conversation", "chat"])
        for speaker, line in chat_lines:
            # Add speaker names as keywords
            keywords.add(speaker.lower())
            # Add key words from the conversation (simplified)
            words = line.lower().split()[:5]
            keywords.update(w for w in words if len(w) > 3)

        # Store in associative memory
        created = persona.scratch.curr_time
        expiration = created + datetime.timedelta(days=30)
        s = persona.scratch.name
        p = "chat with"
        o = partner

        # Calculate poignancy based on conversation length
        poignancy = min(5 + num_lines, 10)

        persona.a_mem.add_chat(
            created=created,
            expiration=expiration,
            s=s,
            p=p,
            o=o,
            description=description,
            keywords=keywords,
            poignancy=poignancy,
            embedding_key=description,
            filling=chat_lines,
        )

        cli.print_info(f"  Stored conversation: {s} <-> {partner} ({num_lines} lines)")

    def start_http_server(self):
        """Start the Flask HTTP server in a background thread."""
        import logging
        import os
        import sys

        # Suppress Flask/Werkzeug output completely
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.disabled = True

        # Redirect Flask's startup message to devnull
        cli_module = sys.modules.get("flask.cli")
        if cli_module:
            cli_module.show_server_banner = lambda *args, **kwargs: None

        def run_flask():
            # Suppress startup messages
            with open(os.devnull, "w") as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    self.flask_app.run(
                        host="127.0.0.1",
                        port=BACKEND_PORT,
                        threaded=True,
                        use_reloader=False,
                    )
                finally:
                    sys.stderr = old_stderr

        self.flask_thread = threading.Thread(target=run_flask, daemon=True)
        self.flask_thread.start()
        cli.print_info(f"HTTP server started on http://127.0.0.1:{BACKEND_PORT}")

    def save(self):
        """
        Save all Reverie progress -- this includes Reverie's global state as well
        as all the personas.

        INPUT
          None
        OUTPUT
          None
          * Saves all relevant data to the designated memory directory
        """
        # <sim_folder> points to the current simulation folder.
        sim_folder = f"{fs_storage}/{self.sim_code}"

        # Save Reverie meta information.
        reverie_meta = dict()
        reverie_meta["fork_sim_code"] = self.fork_sim_code
        reverie_meta["start_date"] = self.start_time.strftime("%B %d, %Y")
        reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
        reverie_meta["sec_per_step"] = self.sec_per_step
        reverie_meta["maze_name"] = self.maze.maze_name
        reverie_meta["persona_names"] = list(self.personas.keys())
        reverie_meta["step"] = self.step
        # Save persona positions directly in meta (avoids needing environment files)
        reverie_meta["persona_tiles"] = {
            name: list(tile) for name, tile in self.personas_tile.items()
        }
        reverie_meta_f = f"{sim_folder}/reverie/meta.json"
        with open(reverie_meta_f, "w") as outfile:
            outfile.write(json.dumps(reverie_meta, indent=2))

        # Save the personas.
        for persona_name, persona in self.personas.items():
            save_folder = f"{sim_folder}/personas/{persona_name}/bootstrap_memory"
            persona.save(save_folder)

    def start_path_tester_server(self):
        """
        Starts the path tester server. This is for generating the spatial memory
        that we need for bootstrapping a persona's state.

        To use this, you need to open server and enter the path tester mode, and
        open the front-end side of the browser.

        INPUT
          None
        OUTPUT
          None
          * Saves the spatial memory of the test agent to the path_tester_env.json
            of the temp storage.
        """

        def print_tree(tree):
            def _print_tree(tree, depth):
                dash = " >" * depth

                if isinstance(tree, list):
                    if tree:
                        print(dash, tree)
                    return

                for key, val in tree.items():
                    if key:
                        print(dash, key)
                    _print_tree(val, depth + 1)

            _print_tree(tree, 0)

        # <curr_vision> is the vision radius of the test agent. Recommend 8 as
        # our default.
        curr_vision = 8
        # <s_mem> is our test spatial memory.
        s_mem = dict()

        # The main while loop for the test agent.
        while True:
            try:
                curr_dict = {}
                tester_file = fs_temp_storage + "/path_tester_env.json"
                if os.path.exists(tester_file):
                    with open(tester_file) as json_file:
                        curr_dict = json.load(json_file)
                        os.remove(tester_file)

                    # Current camera location
                    curr_sts = self.maze.sq_tile_size
                    curr_camera = (
                        int(math.ceil(curr_dict["x"] / curr_sts)),
                        int(math.ceil(curr_dict["y"] / curr_sts)) + 1,
                    )
                    curr_tile_det = self.maze.access_tile(curr_camera)

                    # Initiating the s_mem
                    world = curr_tile_det["world"]
                    if curr_tile_det["world"] not in s_mem:
                        s_mem[world] = dict()

                    # Iterating throughn the nearby tiles.
                    nearby_tiles = self.maze.get_nearby_tiles(curr_camera, curr_vision)
                    for i in nearby_tiles:
                        i_det = self.maze.access_tile(i)
                        if (
                            curr_tile_det["sector"] == i_det["sector"]
                            and curr_tile_det["arena"] == i_det["arena"]
                        ):
                            if i_det["sector"] != "":
                                if i_det["sector"] not in s_mem[world]:
                                    s_mem[world][i_det["sector"]] = dict()
                            if i_det["arena"] != "":
                                if i_det["arena"] not in s_mem[world][i_det["sector"]]:
                                    s_mem[world][i_det["sector"]][
                                        i_det["arena"]
                                    ] = list()
                            if i_det["game_object"] != "":
                                if (
                                    i_det["game_object"]
                                    not in s_mem[world][i_det["sector"]][i_det["arena"]]
                                ):
                                    s_mem[world][i_det["sector"]][i_det["arena"]] += [
                                        i_det["game_object"]
                                    ]

                # Incrementally outputting the s_mem and saving the json file.
                print("= " * 15)
                out_file = fs_temp_storage + "/path_tester_out.json"
                with open(out_file, "w") as outfile:
                    outfile.write(json.dumps(s_mem, indent=2))
                print_tree(s_mem)

            except Exception:
                pass

            time.sleep(self.server_sleep * 10)

    def run_steps(self, num_steps):
        """
        Run the simulation for a given number of steps (CLI mode).

        This runs the cognitive pipeline directly. Movements are queued
        for the frontend to poll and display.

        INPUT
          num_steps: Number of simulation steps to run
        OUTPUT
          None
        """
        for i in range(num_steps):
            # Print step header
            cli.print_step_start(self.step, self.curr_time)

            # Build environment data from current backend state
            environment = {}
            for persona_name in self.personas:
                tile = self.personas_tile[persona_name]
                environment[persona_name] = {"x": tile[0], "y": tile[1]}

            # Run one step - movements are queued for frontend
            self._process_step({"environment": environment})

    def open_server(self):
        """
        Open up an interactive terminal prompt that lets you run the simulation
        step by step and probe agent state.
        """
        # Show simulation info
        cli.print_simulation_started(self.sim_code)
        cli.print_sim_info(
            self.sim_code,
            self.fork_sim_code,
            self.curr_time,
            self.step,
            list(self.personas.keys()),
        )

        sim_folder = f"{fs_storage_runs}/{self.sim_code}"

        while True:
            sim_command = cli.get_prompt()
            if not sim_command:
                continue

            try:
                cmd = sim_command.lower().strip()
                parts = sim_command.split()

                # === CONTROL COMMANDS ===
                if cmd in ["f", "fin", "finish", "save", "exit"]:
                    self.save()
                    if cmd != "save":
                        cli.print_success("Simulation saved. Goodbye!")
                        break
                    else:
                        cli.print_success("Simulation saved.")

                elif cmd == "quit":
                    cli.print_warning("Exiting without saving...")
                    shutil.rmtree(sim_folder)
                    break

                elif cmd.startswith("run"):
                    if len(parts) < 2:
                        cli.print_error("Usage: run <number_of_steps>")
                        continue
                    try:
                        int_count = int(parts[-1])
                        start_time = time.time()
                        self.run_steps(int_count)
                        elapsed = time.time() - start_time
                        cli.print_run_complete(int_count, elapsed)
                    except ValueError:
                        cli.print_error(f"Invalid step count: {parts[-1]}")

                # === STATUS COMMANDS ===
                elif cmd in ["help", "?"]:
                    cli.print_help()

                elif cmd == "status":
                    cli.print_sim_info(
                        self.sim_code,
                        self.fork_sim_code,
                        self.curr_time,
                        self.step,
                        list(self.personas.keys()),
                    )

                elif cmd == "time":
                    cli.print_info(
                        f"Simulation time: {self.curr_time.strftime('%B %d, %Y, %H:%M:%S')}"
                    )
                    cli.print_info(f"Step: {self.step}")

                elif cmd == "personas":
                    for name in self.personas.keys():
                        persona = self.personas[name]
                        action = persona.scratch.act_description or "idle"
                        cli.print_persona_action(name, action)

                # === PERSONA COMMANDS ===
                elif cmd.startswith("schedule "):
                    name = " ".join(parts[1:])
                    if name in self.personas:
                        print(
                            self.personas[name].scratch.get_str_daily_schedule_summary()
                        )
                    else:
                        cli.print_error(f"Persona '{name}' not found")

                elif cmd.startswith("location "):
                    name = " ".join(parts[1:])
                    if name in self.personas:
                        tile = self.personas[name].scratch.curr_tile
                        addr = self.personas[name].scratch.act_address
                        cli.print_info(f"{name} is at tile {tile}")
                        cli.print_info(f"Location: {addr}")
                    else:
                        cli.print_error(f"Persona '{name}' not found")

                elif cmd.startswith("memory "):
                    name = " ".join(parts[1:])
                    if name in self.personas:
                        p = self.personas[name]
                        cli.print_memory_summary(
                            name,
                            len(p.a_mem.seq_event)
                            if hasattr(p.a_mem, "seq_event")
                            else 0,
                            len(p.a_mem.seq_thought)
                            if hasattr(p.a_mem, "seq_thought")
                            else 0,
                            len(p.a_mem.seq_chat)
                            if hasattr(p.a_mem, "seq_chat")
                            else 0,
                        )
                    else:
                        cli.print_error(f"Persona '{name}' not found")

                elif cmd.startswith("chat "):
                    name = " ".join(parts[1:])
                    if name in self.personas:
                        self.personas[name].open_convo_session("analysis")
                    else:
                        cli.print_error(f"Persona '{name}' not found")

                # === LEGACY COMMANDS (for backwards compatibility) ===
                elif "print persona schedule" in cmd:
                    name = " ".join(parts[-2:])
                    if name in self.personas:
                        print(
                            self.personas[name].scratch.get_str_daily_schedule_summary()
                        )

                elif "print all persona schedule" in cmd:
                    for persona_name, persona in self.personas.items():
                        print(f"\n{persona_name}")
                        print(persona.scratch.get_str_daily_schedule_summary())
                        print("---")

                elif "print current time" in cmd:
                    print(f"{self.curr_time.strftime('%B %d, %Y, %H:%M:%S')}")
                    print(f"steps: {self.step}")

                elif "call -- analysis" in cmd:
                    persona_name = sim_command[len("call -- analysis") :].strip()
                    if persona_name in self.personas:
                        self.personas[persona_name].open_convo_session("analysis")

                elif cmd == "start path tester mode":
                    shutil.rmtree(sim_folder)
                    self.start_path_tester_server()

                else:
                    cli.print_warning(f"Unknown command: {sim_command}")
                    cli.print_info("Type 'help' for available commands")

            except KeyboardInterrupt:
                print()
                continue
            except Exception as e:
                cli.print_error(str(e))
                if debug:
                    traceback.print_exc()


def load_local_config():
    """Load local config from project root (two levels up from this file)."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "local_config.json"
    )
    config_path = os.path.normpath(config_path)
    try:
        with open(config_path, "r") as f:
            return json.load(f), config_path
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return {
            "default_fork": "base_the_ville_isabella_maria_klaus",
            "last_simulation": None,
        }, config_path


def save_local_config(config, config_path):
    """Save local config to project root."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def generate_simulation_name(fork_name):
    """Generate a new simulation name based on fork name + timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{fork_name}_{timestamp}"


if __name__ == "__main__":
    import sys

    # Load local config
    config, config_path = load_local_config()
    default_fork = config.get("default_fork", "the_ville_isabella_maria_klaus")
    last_sim = config.get("last_simulation")

    # Show startup menu
    cli.print_startup_menu(default_fork, last_sim)

    choice = input(cli.c("  > ", cli.Colors.BRIGHT_BLACK)).strip().lower()

    if choice in ["c", "continue"]:
        if not last_sim:
            cli.print_error("No previous simulation to continue.")
            sys.exit(1)
        target = last_sim
        target_folder = f"{fs_storage_runs}/{target}"
        if not os.path.exists(target_folder):
            cli.print_error(f"Simulation '{target}' not found.")
            sys.exit(1)
        # Load the existing simulation's meta to get its fork
        meta_file = f"{target_folder}/reverie/meta.json"
        with open(meta_file) as json_file:
            meta = json.load(json_file)
        origin = meta.get("fork_sim_code", default_fork)
        cli.print_info(f"Continuing: {target}")

    elif choice == "custom":
        # Prompt for fork simulation with default
        print(f"\n  Fork from [{cli.c(default_fork, cli.Colors.CYAN)}]: ", end="")
        origin = input().strip()
        if not origin:
            origin = default_fork

        # Prompt for target simulation with auto-generated default
        auto_target = generate_simulation_name(origin)
        print(f"  New name [{cli.c(auto_target, cli.Colors.GREEN)}]: ", end="")
        target = input().strip()
        if not target:
            target = auto_target

    else:
        # Default: start new simulation with auto-generated name
        origin = default_fork
        target = generate_simulation_name(origin)

    # Save the simulation name to local config
    config["last_simulation"] = target
    save_local_config(config, config_path)

    rs = ReverieServer(origin, target)

    # Start HTTP server for frontend communication
    rs.start_http_server()

    # Run CLI interface
    rs.open_server()
