"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: print_prompt.py
Description: For printing prompts when the setting for verbose is set to True.
"""
import sys
sys.path.append('../')

import json
import numpy
import datetime
import random
import os

from global_methods import *
from persona.prompt_template.claude_structure import *
from utils import *

# Import cli_interface if available for colored output
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    import cli_interface as cli
    HAS_CLI = True
except ImportError:
    HAS_CLI = False

##############################################################################
#                    PERSONA Chapter 1: Prompt Structures                    #
##############################################################################

def print_run_prompts(prompt_template=None,
                      persona=None,
                      prompt_input=None,
                      prompt=None,
                      output=None):
  """Print prompt debug info in a clean, compact format."""

  # Extract just the function name from the template path
  func_name = prompt_template.split('/')[-1].replace('.txt', '') if prompt_template else 'unknown'

  if HAS_CLI:
    # Compact colored output
    print(cli.c(f"  ⚙ {func_name}", cli.Colors.DIM) +
          cli.c(f" → ", cli.Colors.BRIGHT_BLACK) +
          cli.c(_truncate_output(output), cli.Colors.WHITE))
  else:
    # Fallback: simple one-liner
    print(f"  [{func_name}] → {_truncate_output(output)}")


def _truncate_output(output, max_len=80):
  """Truncate output for display."""
  if output is None:
    return "None"
  s = str(output).replace('\n', ' ').strip()
  if len(s) > max_len:
    return s[:max_len] + "..."
  return s


def print_run_prompts_verbose(prompt_template=None,
                              persona=None,
                              prompt_input=None,
                              prompt=None,
                              output=None):
  """Full verbose output - call this explicitly when debugging."""
  print (f"=== {prompt_template}")
  print ("~~~ persona    ---------------------------------------------------")
  print (persona.name, "\n")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n")
  print ("=== END ==========================================================")
  print ("\n\n\n")
