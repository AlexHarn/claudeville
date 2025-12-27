"""
Original Author: Joon Sung Park (joonspk@stanford.edu)
Heavily modified for Claudeville (Claude CLI port)

File: reverie.py
Description: Main program for running generative agent simulations.
"""

import datetime
import json
import math
import os
import shutil
import threading
import time
import traceback

from flask import Flask, jsonify

import cli_interface as cli
from maze import Maze
from utils import (
    debug,
    fs_storage,
    fs_storage_base,
    fs_storage_runs,
    fs_temp_storage,
)

from persona.persona import Persona

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

        # Run cognitive pipeline for each persona
        movements = {"persona": {}, "meta": {}}
        for persona_name, persona in self.personas.items():
            next_tile, pronunciatio, description = persona.move(
                self.maze,
                self.personas,
                self.personas_tile[persona_name],
                self.curr_time,
            )
            movements["persona"][persona_name] = {
                "movement": next_tile,
                "pronunciatio": pronunciatio,
                "description": description,
                "chat": persona.scratch.chat,
            }

        # Add meta information (step is sent BEFORE increment so frontend knows what step this was)
        movements["meta"]["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
        movements["meta"]["step"] = self.step  # Current step being processed

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

    def start_http_server(self):
        """Start the Flask HTTP server in a background thread."""
        import logging

        # Suppress Flask's default logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)

        def run_flask():
            self.flask_app.run(
                host="127.0.0.1",
                port=BACKEND_PORT,
                threaded=True,
                use_reloader=False,
            )

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
        for _ in range(num_steps):
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
