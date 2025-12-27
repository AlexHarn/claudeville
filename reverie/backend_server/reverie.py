"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modified for Claudeville

File: reverie.py
Description: Main program for running generative agent simulations.
"""
import json
import numpy
import datetime
import pickle
import time
import math
import os
import shutil
import traceback

from selenium import webdriver

from global_methods import *
from utils import *
from maze import *
from persona.persona import *
import cli_interface as cli

##############################################################################
#                                  REVERIE                                   #
##############################################################################

class ReverieServer:
  def __init__(self,
               fork_sim_code,
               sim_code):
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
    copyanything(fork_folder, sim_folder)

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
                        f"{reverie_meta['start_date']}, 00:00:00",  
                        "%B %d, %Y, %H:%M:%S")
    # <curr_time> is the datetime instance that indicates the game's current
    # time. This gets incremented by <sec_per_step> amount everytime the world
    # progresses (that is, everytime curr_env_file is recieved). 
    self.curr_time = datetime.datetime.strptime(reverie_meta['curr_time'], 
                                                "%B %d, %Y, %H:%M:%S")
    # <sec_per_step> denotes the number of seconds in game time that each 
    # step moves foward. 
    self.sec_per_step = reverie_meta['sec_per_step']
    
    # <maze> is the main Maze instance. Note that we pass in the maze_name
    # (e.g., "double_studio") to instantiate Maze. 
    # e.g., Maze("double_studio")
    self.maze = Maze(reverie_meta['maze_name'])
    
    # <step> denotes the number of steps that our game has taken. A step here
    # literally translates to the number of moves our personas made in terms
    # of the number of tiles. 
    self.step = reverie_meta['step']

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
    init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
    init_env = json.load(open(init_env_file))
    for persona_name in reverie_meta['persona_names']: 
      persona_folder = f"{sim_folder}/personas/{persona_name}"
      p_x = init_env[persona_name]["x"]
      p_y = init_env[persona_name]["y"]
      curr_persona = Persona(persona_name, persona_folder)

      self.personas[persona_name] = curr_persona
      self.personas_tile[persona_name] = (p_x, p_y)
      self.maze.tiles[p_y][p_x]["events"].add(curr_persona.scratch
                                              .get_curr_event_and_desc())

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

        if type(tree) == type(list()): 
          if tree:
            print (dash, tree)
          return 

        for key, val in tree.items(): 
          if key: 
            print (dash, key)
          _print_tree(val, depth+1)
      
      _print_tree(tree, 0)

    # <curr_vision> is the vision radius of the test agent. Recommend 8 as 
    # our default. 
    curr_vision = 8
    # <s_mem> is our test spatial memory. 
    s_mem = dict()

    # The main while loop for the test agent. 
    while (True): 
      try: 
        curr_dict = {}
        tester_file = fs_temp_storage + "/path_tester_env.json"
        if check_if_file_exists(tester_file): 
          with open(tester_file) as json_file: 
            curr_dict = json.load(json_file)
            os.remove(tester_file)
          
          # Current camera location
          curr_sts = self.maze.sq_tile_size
          curr_camera = (int(math.ceil(curr_dict["x"]/curr_sts)), 
                         int(math.ceil(curr_dict["y"]/curr_sts))+1)
          curr_tile_det = self.maze.access_tile(curr_camera)

          # Initiating the s_mem
          world = curr_tile_det["world"]
          if curr_tile_det["world"] not in s_mem: 
            s_mem[world] = dict()

          # Iterating throughn the nearby tiles.
          nearby_tiles = self.maze.get_nearby_tiles(curr_camera, curr_vision)
          for i in nearby_tiles: 
            i_det = self.maze.access_tile(i)
            if (curr_tile_det["sector"] == i_det["sector"] 
                and curr_tile_det["arena"] == i_det["arena"]): 
              if i_det["sector"] != "": 
                if i_det["sector"] not in s_mem[world]: 
                  s_mem[world][i_det["sector"]] = dict()
              if i_det["arena"] != "": 
                if i_det["arena"] not in s_mem[world][i_det["sector"]]: 
                  s_mem[world][i_det["sector"]][i_det["arena"]] = list()
              if i_det["game_object"] != "": 
                if (i_det["game_object"] 
                    not in s_mem[world][i_det["sector"]][i_det["arena"]]):
                  s_mem[world][i_det["sector"]][i_det["arena"]] += [
                                                         i_det["game_object"]]

        # Incrementally outputting the s_mem and saving the json file. 
        print ("= " * 15)
        out_file = fs_temp_storage + "/path_tester_out.json"
        with open(out_file, "w") as outfile: 
          outfile.write(json.dumps(s_mem, indent=2))
        print_tree(s_mem)

      except:
        pass

      time.sleep(self.server_sleep * 10)


  def start_server(self, int_counter): 
    """
    The main backend server of Reverie. 
    This function retrieves the environment file from the frontend to 
    understand the state of the world, calls on each personas to make 
    decisions based on the world state, and saves their moves at certain step
    intervals. 
    INPUT
      int_counter: Integer value for the number of steps left for us to take
                   in this iteration. 
    OUTPUT 
      None
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # When a persona arrives at a game object, we give a unique event
    # to that object. 
    # e.g., ('double studio[...]:bed', 'is', 'unmade', 'unmade')
    # Later on, before this cycle ends, we need to return that to its 
    # initial state, like this: 
    # e.g., ('double studio[...]:bed', None, None, None)
    # So we need to keep track of which event we added. 
    # <game_obj_cleanup> is used for that. 
    game_obj_cleanup = dict()

    # The main while loop of Reverie. 
    while (True): 
      # Done with this iteration if <int_counter> reaches 0. 
      if int_counter == 0: 
        break

      # <curr_env_file> file is the file that our frontend outputs. When the
      # frontend has done its job and moved the personas, then it will put a 
      # new environment file that matches our step count. That's when we run 
      # the content of this for loop. Otherwise, we just wait. 
      curr_env_file = f"{sim_folder}/environment/{self.step}.json"
      if check_if_file_exists(curr_env_file):
        # If we have an environment file, it means we have a new perception
        # input to our personas. So we first retrieve it.
        try: 
          # Try and save block for robustness of the while loop.
          with open(curr_env_file) as json_file:
            new_env = json.load(json_file)
            env_retrieved = True
        except: 
          pass
      
        if env_retrieved: 
          # This is where we go through <game_obj_cleanup> to clean up all 
          # object actions that were used in this cylce. 
          for key, val in game_obj_cleanup.items(): 
            # We turn all object actions to their blank form (with None). 
            self.maze.turn_event_from_tile_idle(key, val)
          # Then we initialize game_obj_cleanup for this cycle. 
          game_obj_cleanup = dict()

          # We first move our personas in the backend environment to match 
          # the frontend environment. 
          for persona_name, persona in self.personas.items(): 
            # <curr_tile> is the tile that the persona was at previously. 
            curr_tile = self.personas_tile[persona_name]
            # <new_tile> is the tile that the persona will move to right now,
            # during this cycle. 
            new_tile = (new_env[persona_name]["x"], 
                        new_env[persona_name]["y"])

            # We actually move the persona on the backend tile map here. 
            self.personas_tile[persona_name] = new_tile
            self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
            self.maze.add_event_from_tile(persona.scratch
                                         .get_curr_event_and_desc(), new_tile)

            # Now, the persona will travel to get to their destination. *Once*
            # the persona gets there, we activate the object action.
            if not persona.scratch.planned_path: 
              # We add that new object action event to the backend tile map. 
              # At its creation, it is stored in the persona's backend. 
              game_obj_cleanup[persona.scratch
                               .get_curr_obj_event_and_desc()] = new_tile
              self.maze.add_event_from_tile(persona.scratch
                                     .get_curr_obj_event_and_desc(), new_tile)
              # We also need to remove the temporary blank action for the 
              # object that is currently taking the action. 
              blank = (persona.scratch.get_curr_obj_event_and_desc()[0], 
                       None, None, None)
              self.maze.remove_event_from_tile(blank, new_tile)

          # Then we need to actually have each of the personas perceive and
          # move. The movement for each of the personas comes in the form of
          # x y coordinates where the persona will move towards. e.g., (50, 34)
          # This is where the core brains of the personas are invoked. 
          movements = {"persona": dict(), 
                       "meta": dict()}
          for persona_name, persona in self.personas.items(): 
            # <next_tile> is a x,y coordinate. e.g., (58, 9)
            # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
            # <description> is a string description of the movement. e.g., 
            #   writing her next novel (editing her novel) 
            #   @ double studio:double studio:common room:sofa
            next_tile, pronunciatio, description = persona.move(
              self.maze, self.personas, self.personas_tile[persona_name], 
              self.curr_time)
            movements["persona"][persona_name] = {}
            movements["persona"][persona_name]["movement"] = next_tile
            movements["persona"][persona_name]["pronunciatio"] = pronunciatio
            movements["persona"][persona_name]["description"] = description
            movements["persona"][persona_name]["chat"] = (persona
                                                          .scratch.chat)

          # Include the meta information about the current stage in the 
          # movements dictionary. 
          movements["meta"]["curr_time"] = (self.curr_time 
                                             .strftime("%B %d, %Y, %H:%M:%S"))

          # We then write the personas' movements to a file that will be sent 
          # to the frontend server. 
          # Example json output: 
          # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
          #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
          #  "meta": {curr_time: <datetime>}}
          curr_move_file = f"{sim_folder}/movement/{self.step}.json"
          with open(curr_move_file, "w") as outfile: 
            outfile.write(json.dumps(movements, indent=2))

          # After this cycle, the world takes one step forward, and the
          # current time moves by <sec_per_step> amount.
          self.step += 1
          self.curr_time += datetime.timedelta(seconds=self.sec_per_step)

          # Signal the frontend of current step (for browser refresh recovery)
          with open(f"{fs_temp_storage}/curr_step.json", "w") as outfile:
            outfile.write(json.dumps({"step": self.step}, indent=2))

          int_counter -= 1
          
      # Sleep so we don't burn our machines. 
      time.sleep(self.server_sleep)


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
      list(self.personas.keys())
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
            self.start_server(int_count)
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
            list(self.personas.keys())
          )

        elif cmd == "time":
          cli.print_info(f"Simulation time: {self.curr_time.strftime('%B %d, %Y, %H:%M:%S')}")
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
            print(self.personas[name].scratch.get_str_daily_schedule_summary())
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
              len(p.a_mem.seq_event) if hasattr(p.a_mem, 'seq_event') else 0,
              len(p.a_mem.seq_thought) if hasattr(p.a_mem, 'seq_thought') else 0,
              len(p.a_mem.seq_chat) if hasattr(p.a_mem, 'seq_chat') else 0
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
            print(self.personas[name].scratch.get_str_daily_schedule_summary())

        elif "print all persona schedule" in cmd:
          for persona_name, persona in self.personas.items():
            print(f"\n{persona_name}")
            print(persona.scratch.get_str_daily_schedule_summary())
            print("---")

        elif "print current time" in cmd:
          print(f'{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}')
          print(f'steps: {self.step}')

        elif "call -- analysis" in cmd:
          persona_name = sim_command[len("call -- analysis"):].strip()
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
  config_path = os.path.join(os.path.dirname(__file__), "..", "..", "local_config.json")
  config_path = os.path.normpath(config_path)
  try:
    with open(config_path, "r") as f:
      return json.load(f), config_path
  except FileNotFoundError:
    # Return default config if file doesn't exist
    return {"default_fork": "base_the_ville_isabella_maria_klaus", "last_simulation": None}, config_path


def save_local_config(config, config_path):
  """Save local config to project root."""
  with open(config_path, "w") as f:
    json.dump(config, f, indent=2)


def generate_simulation_name(fork_name):
  """Generate a new simulation name based on fork name + timestamp."""
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  return f"{fork_name}_{timestamp}"


if __name__ == '__main__':
  import sys

  # Load local config
  config, config_path = load_local_config()
  default_fork = config.get("default_fork", "the_ville_isabella_maria_klaus")
  last_sim = config.get("last_simulation")

  # Show startup menu
  cli.print_startup_menu(default_fork, last_sim)

  choice = input(cli.c("  > ", cli.Colors.BRIGHT_BLACK)).strip().lower()

  if choice in ['c', 'continue']:
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

  elif choice == 'custom':
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
  rs.open_server()




















































