"""
Claude CLI Wrapper for Claudeville

This module replaces the OpenAI API wrapper (gpt_structure.py) with Claude CLI
commands for the Claudeville project. It provides equivalent functionality using
the Claude CLI with Max subscription.

Key differences from the original:
- Uses subprocess to call Claude CLI instead of OpenAI API
- No embedding functions (embeddings removed entirely in Claudeville)
- Sonnet model replaces GPT-3.5-turbo
- Opus model replaces GPT-4

Author: Claudeville Project
Original: Joon Sung Park (joonspk@stanford.edu)
"""

import json
import subprocess
import time
import re


SYSTEM_PROMPT = """You are a response generator for a simulation.
CRITICAL: Output ONLY the exact requested format. No explanations, no preamble, no "I see" or "Based on", no conversational text.
If asked for JSON, output ONLY valid JSON.
If asked to complete a sentence, output ONLY the completion.
If asked for a time, output ONLY the time (e.g., "7:00 AM").
If asked for emojis, output ONLY the emojis.
Never explain your reasoning. Never acknowledge the prompt. Just output the answer."""


def temp_sleep(seconds=0.1):
    """Brief pause between API calls to avoid rate limiting."""
    time.sleep(seconds)


def _run_claude_cli(prompt, model="sonnet", system_prompt=None):
    """
    Execute Claude CLI and return the result.

    ARGS:
        prompt: The prompt string to send to Claude
        model: The model to use ("sonnet" or "opus")
        system_prompt: Optional system prompt to override the default SYSTEM_PROMPT

    RETURNS:
        The text response from Claude

    RAISES:
        subprocess.CalledProcessError: If CLI execution fails
        json.JSONDecodeError: If response parsing fails
    """
    cmd = ["claude", "-p", "--output-format", "json", "--model", model]

    # Always use a system prompt - default to SYSTEM_PROMPT if none provided
    effective_system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
    cmd.extend(["--system-prompt", effective_system_prompt])

    cmd.append(prompt)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "Unknown CLI error"
        raise subprocess.CalledProcessError(result.returncode, cmd, error_msg)

    try:
        response_data = json.loads(result.stdout)
        return response_data.get("result", "")
    except json.JSONDecodeError:
        # If JSON parsing fails, return raw stdout as fallback
        return result.stdout.strip()


# ============================================================================
# #####################[SECTION 1: CLAUDE STRUCTURE] #########################
# ============================================================================

def Claude_single_request(prompt):
    """
    Make a single request to Claude using Sonnet model.

    ARGS:
        prompt: a str prompt

    RETURNS:
        a str of Claude's response.
    """
    temp_sleep()
    return _run_claude_cli(prompt, model="sonnet")


def Claude_opus_request(prompt):
    """
    Make a request to Claude using the Opus model (most capable).

    ARGS:
        prompt: a str prompt

    RETURNS:
        a str of Claude's response.
    """
    temp_sleep()

    try:
        return _run_claude_cli(prompt, model="opus")
    except Exception as e:
        print(f"Claude Opus ERROR: {e}")
        return "Claude Opus ERROR"


def Claude_sonnet_request(prompt):
    """
    Make a request to Claude using the Sonnet model (fast and efficient).

    ARGS:
        prompt: a str prompt

    RETURNS:
        a str of Claude's response.
    """
    try:
        return _run_claude_cli(prompt, model="sonnet")
    except Exception as e:
        print(f"Claude Sonnet ERROR: {e}")
        return "Claude Sonnet ERROR"


def Claude_opus_safe_generate_response(prompt,
                                       example_output,
                                       special_instruction,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):  # verbose kept for signature compat, but ignored
    """
    Generate a response using Opus with validation and retry logic.

    ARGS:
        prompt: The base prompt
        example_output: Example of expected output format
        special_instruction: Additional instructions for formatting
        repeat: Number of retry attempts
        fail_safe_response: Response to return if all attempts fail
        func_validate: Validation function for the response
        func_clean_up: Cleanup function for the response
        verbose: Ignored (kept for signature compatibility)

    RETURNS:
        Cleaned response if valid, fail_safe_response otherwise
    """
    prompt = 'Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    for i in range(repeat):
        try:
            curr_response = Claude_opus_request(prompt).strip()
            end_index = curr_response.rfind('}') + 1
            curr_response = curr_response[:end_index]
            curr_response = json.loads(curr_response)["output"]

            if func_validate(curr_response, prompt=prompt):
                return func_clean_up(curr_response, prompt=prompt)

        except Exception:
            pass

    # All attempts failed - return the fail_safe_response
    return fail_safe_response


def Claude_safe_generate_response(prompt,
                                  example_output,
                                  special_instruction,
                                  repeat=3,
                                  fail_safe_response="error",
                                  func_validate=None,
                                  func_clean_up=None,
                                  verbose=False):  # verbose kept for signature compat, but ignored
    """
    Generate a response using Sonnet with validation and retry logic.

    ARGS:
        prompt: The base prompt
        example_output: Example of expected output format
        special_instruction: Additional instructions for formatting
        repeat: Number of retry attempts
        fail_safe_response: Response to return if all attempts fail
        func_validate: Validation function for the response
        func_clean_up: Cleanup function for the response
        verbose: Ignored (kept for signature compatibility)

    RETURNS:
        Cleaned response if valid, fail_safe_response otherwise
    """
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    for i in range(repeat):
        try:
            curr_response = Claude_sonnet_request(prompt).strip()
            end_index = curr_response.rfind('}') + 1
            curr_response = curr_response[:end_index]
            curr_response = json.loads(curr_response)["output"]

            if func_validate(curr_response, prompt=prompt):
                return func_clean_up(curr_response, prompt=prompt)

        except Exception:
            pass

    # All attempts failed - return the fail_safe_response
    return fail_safe_response


# ============================================================================
# ###################[SECTION 2: PROMPT GENERATION] ##########################
# ============================================================================

def generate_prompt(curr_input, prompt_lib_file):
    """
    Takes in the current input (e.g. comment that you want to classify) and
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this
    function replaces this substr with the actual curr_input to produce the
    final prompt that will be sent to Claude.

    ARGS:
        curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                    INPUT, THIS CAN BE A LIST.)
        prompt_lib_file: the path to the prompt file.

    RETURNS:
        a str prompt that will be sent to Claude.
    """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    with open(prompt_lib_file, "r") as f:
        prompt = f.read()

    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)

    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

    return prompt.strip()


def safe_generate_response(prompt,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
    """
    Generate a response with validation and retry logic.

    ARGS:
        prompt: The prompt string
        repeat: Number of retry attempts
        fail_safe_response: Response to return if all attempts fail
        func_validate: Validation function for the response
        func_clean_up: Cleanup function for the response
        verbose: Whether to print debug information

    RETURNS:
        Cleaned response if valid, fail_safe_response otherwise
    """
    if verbose:
        print(prompt)

    for i in range(repeat):
        try:
            curr_response = Claude_sonnet_request(prompt)
            if func_validate(curr_response, prompt=prompt):
                return func_clean_up(curr_response, prompt=prompt)
            if verbose:
                print("---- repeat count: ", i, curr_response)
                print(curr_response)
                print("~~~~")
        except Exception as e:
            if verbose:
                print(f"Attempt {i} failed: {e}")

    return fail_safe_response


# ============================================================================
# ###################[SECTION 3: BACKWARD COMPATIBILITY] #####################
# ============================================================================

# Drop-in compatibility aliases for code that still references OpenAI functions
ChatGPT_single_request = Claude_single_request
GPT4_request = Claude_opus_request
ChatGPT_request = Claude_sonnet_request
ChatGPT_safe_generate_response = Claude_safe_generate_response
ChatGPT_safe_generate_response_OLD = Claude_safe_generate_response
GPT4_safe_generate_response = Claude_opus_safe_generate_response


# ============================================================================
# ###################[SECTION 4: TESTING] ####################################
# ============================================================================

if __name__ == '__main__':
    # Test basic functionality
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"

    try:
        prompt = generate_prompt(curr_input, prompt_lib_file)

        def __func_validate(response, prompt=None):
            if len(response.strip()) <= 1:
                return False
            if len(response.strip().split(" ")) > 1:
                return False
            return True

        def __func_clean_up(response, prompt=None):
            cleaned_response = response.strip()
            return cleaned_response

        output = safe_generate_response(prompt,
                                        None,  # gpt_parameter ignored
                                        5,
                                        "rest",
                                        __func_validate,
                                        __func_clean_up,
                                        True)

        print(output)

    except FileNotFoundError:
        print("Test prompt file not found. Testing basic Claude request...")
        response = Claude_single_request("Say 'Hello, Claudeville!' in exactly those words.")
        print(f"Response: {response}")
