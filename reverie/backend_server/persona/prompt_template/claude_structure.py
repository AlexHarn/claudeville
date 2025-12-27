"""
Claude Agent SDK Wrapper for Claudeville

This module replaces the OpenAI API wrapper (gpt_structure.py) with the Claude Agent SDK
for the Claudeville project. It provides equivalent functionality using the SDK which
leverages existing Claude Code authentication (Max subscription).

Key features:
- Uses ClaudeSDKClient for persistent connections (~2.5s per call vs ~7-10s)
- Async-based with sync wrappers for compatibility
- Automatic context monitoring with 80% threshold compaction
- Session support for persistent persona contexts
- No embedding functions (embeddings removed entirely in Claudeville)
- Sonnet model replaces GPT-3.5-turbo
- Opus model replaces GPT-4

Author: Claudeville Project
Original: Joon Sung Park (joonspk@stanford.edu)
"""

import asyncio
import atexit
import json
import threading
import time
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import ResultMessage

# Context window limits
MAX_CONTEXT_TOKENS = 200000  # Claude's context window
COMPACTION_THRESHOLD = 0.80  # Trigger compaction at 80% fill
COMPACTION_TOKEN_LIMIT = int(MAX_CONTEXT_TOKENS * COMPACTION_THRESHOLD)  # 160K tokens

SYSTEM_PROMPT = """You are a response generator for a simulation.
CRITICAL: Output ONLY the exact requested format. No explanations, no preamble, no "I see" or "Based on", no conversational text.
If asked for JSON, output ONLY valid JSON.
If asked to complete a sentence, output ONLY the completion.
If asked for a time, output ONLY the time (e.g., "7:00 AM").
If asked for emojis, output ONLY the emojis.
Never explain your reasoning. Never acknowledge the prompt. Just output the answer."""


# ============================================================================
# #####################[SECTION 1: CLIENT MANAGEMENT] ########################
# ============================================================================

# Global client pool - maps model names to persistent clients
_client_pool: dict[str, ClaudeSDKClient] = {}
_client_locks: dict[str, asyncio.Lock] = {}
_client_usage: dict[str, dict[str, Any]] = {}  # Track token usage per client

# Persistent event loop running in a background thread
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


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

            # Register cleanup on exit
            atexit.register(_shutdown_loop)

    return _loop


def _shutdown_loop():
    """Shutdown the background event loop."""
    global _loop, _loop_thread

    if _loop is not None and _loop.is_running():
        # Schedule cleanup of clients
        future = asyncio.run_coroutine_threadsafe(cleanup_clients(), _loop)
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


def _get_context_tokens(usage: dict[str, Any] | None) -> int:
    """Calculate total context tokens from usage stats."""
    if not usage:
        return 0
    return (
        usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("input_tokens", 0)
    )


async def _get_or_create_client(
    model: str, system_prompt: str | None = None
) -> ClaudeSDKClient:
    """
    Get an existing client or create a new one for the given model.

    Clients are pooled by model name to reuse connections.
    """
    effective_system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
    client_key = f"{model}:{hash(effective_system_prompt)}"

    # Create lock for this client key if needed
    if client_key not in _client_locks:
        _client_locks[client_key] = asyncio.Lock()

    async with _client_locks[client_key]:
        # Check if we need to create or recreate the client
        if client_key not in _client_pool:
            options = ClaudeAgentOptions(
                allowed_tools=[],  # No tools for simple prompts
                permission_mode="bypassPermissions",
                system_prompt=effective_system_prompt,
                model=model,
            )
            client = ClaudeSDKClient(options)
            await client.connect()
            _client_pool[client_key] = client
            _client_usage[client_key] = {"context_tokens": 0}

        return _client_pool[client_key]


async def _check_and_handle_compaction(
    client_key: str, usage: dict[str, Any] | None
) -> bool:
    """
    Check if compaction is needed based on token usage.

    Returns True if compaction was triggered (client needs to be recreated).
    """
    context_tokens = _get_context_tokens(usage)
    _client_usage[client_key] = {"context_tokens": context_tokens}

    if context_tokens >= COMPACTION_TOKEN_LIMIT:
        print(
            f"[Claudeville] Context at {context_tokens}/{MAX_CONTEXT_TOKENS} tokens ({context_tokens/MAX_CONTEXT_TOKENS*100:.1f}%) - triggering compaction"
        )

        # Disconnect the old client
        if client_key in _client_pool:
            try:
                await _client_pool[client_key].disconnect()
            except Exception:
                pass
            del _client_pool[client_key]

        return True

    return False


async def _query_with_client(
    prompt: str,
    model: str = "sonnet",
    system_prompt: str | None = None,
) -> str:
    """
    Execute a query using a persistent ClaudeSDKClient.

    This maintains the connection for faster subsequent calls (~2.5s vs ~10s).
    Automatically handles compaction when context window fills.
    """
    effective_system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
    client_key = f"{model}:{hash(effective_system_prompt)}"

    # Get or create client
    client = await _get_or_create_client(model, system_prompt)

    # Send query
    await client.query(prompt)

    # Receive response
    result_text = ""
    usage = None

    async for message in client.receive_response():
        if isinstance(message, ResultMessage):
            result_text = message.result or ""
            usage = message.usage

    # Check if we need to trigger compaction
    await _check_and_handle_compaction(client_key, usage)

    return result_text


def temp_sleep(seconds=0.1):
    """Brief pause between API calls to avoid rate limiting."""
    time.sleep(seconds)


def _run_claude_sdk(
    prompt: str, model: str = "sonnet", system_prompt: str | None = None
) -> str:
    """
    Sync wrapper for Claude Agent SDK query using persistent client.

    Uses a background event loop to maintain persistent ClaudeSDKClient connections
    across calls, enabling ~2.5s per call instead of ~10s.

    ARGS:
        prompt: The prompt string to send to Claude
        model: The model to use ("sonnet" or "opus")
        system_prompt: Optional system prompt to override the default SYSTEM_PROMPT

    RETURNS:
        The text response from Claude
    """
    return _run_async(_query_with_client(prompt, model, system_prompt))


# ============================================================================
# #####################[SECTION 2: CLAUDE STRUCTURE] #########################
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
    return _run_claude_sdk(prompt, model="sonnet")


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
        return _run_claude_sdk(prompt, model="opus")
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
        return _run_claude_sdk(prompt, model="sonnet")
    except Exception as e:
        print(f"Claude Sonnet ERROR: {e}")
        return "Claude Sonnet ERROR"


def Claude_opus_safe_generate_response(
    prompt,
    example_output,
    special_instruction,
    repeat=3,
    fail_safe_response="error",
    func_validate=None,
    func_clean_up=None,
    verbose=False,
):  # verbose kept for signature compat, but ignored
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
    prompt += (
        f"Output the response to the prompt above in json. {special_instruction}\n"
    )
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    for i in range(repeat):
        try:
            curr_response = Claude_opus_request(prompt).strip()
            end_index = curr_response.rfind("}") + 1
            curr_response = curr_response[:end_index]
            curr_response = json.loads(curr_response)["output"]

            if func_validate(curr_response, prompt=prompt):
                return func_clean_up(curr_response, prompt=prompt)

        except Exception:
            pass

    # All attempts failed - return the fail_safe_response
    return fail_safe_response


def Claude_safe_generate_response(
    prompt,
    example_output,
    special_instruction,
    repeat=3,
    fail_safe_response="error",
    func_validate=None,
    func_clean_up=None,
    verbose=False,
):  # verbose kept for signature compat, but ignored
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
    prompt += (
        f"Output the response to the prompt above in json. {special_instruction}\n"
    )
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    for i in range(repeat):
        try:
            curr_response = Claude_sonnet_request(prompt).strip()
            end_index = curr_response.rfind("}") + 1
            curr_response = curr_response[:end_index]
            curr_response = json.loads(curr_response)["output"]

            if func_validate(curr_response, prompt=prompt):
                return func_clean_up(curr_response, prompt=prompt)

        except Exception:
            pass

    # All attempts failed - return the fail_safe_response
    return fail_safe_response


# ============================================================================
# ###################[SECTION 3: PROMPT GENERATION] ##########################
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


def safe_generate_response(
    prompt,
    repeat=5,
    fail_safe_response="error",
    func_validate=None,
    func_clean_up=None,
    verbose=False,
):
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
# ###################[SECTION 4: UTILITY FUNCTIONS] ##########################
# ============================================================================


async def cleanup_clients():
    """
    Cleanup all persistent clients. Call this when shutting down the simulation.
    """
    for client_key, client in list(_client_pool.items()):
        try:
            await client.disconnect()
        except Exception:
            pass
    _client_pool.clear()
    _client_usage.clear()


def cleanup_clients_sync():
    """Sync wrapper for cleanup_clients."""
    if _loop is not None and _loop.is_running():
        _run_async(cleanup_clients())
    else:
        # No loop running, nothing to clean up
        pass


def get_client_stats() -> dict[str, Any]:
    """
    Get statistics about the current client pool.

    RETURNS:
        Dictionary with client stats including context token usage.
    """
    return {
        "num_clients": len(_client_pool),
        "clients": {
            key: {
                "context_tokens": _client_usage.get(key, {}).get("context_tokens", 0),
                "context_pct": _client_usage.get(key, {}).get("context_tokens", 0)
                / MAX_CONTEXT_TOKENS
                * 100,
            }
            for key in _client_pool.keys()
        },
        "compaction_threshold_pct": COMPACTION_THRESHOLD * 100,
        "compaction_token_limit": COMPACTION_TOKEN_LIMIT,
    }


# ============================================================================
# ###################[SECTION 5: BACKWARD COMPATIBILITY] #####################
# ============================================================================

# Drop-in compatibility aliases for code that still references OpenAI functions
ChatGPT_single_request = Claude_single_request
GPT4_request = Claude_opus_request
ChatGPT_request = Claude_sonnet_request
ChatGPT_safe_generate_response = Claude_safe_generate_response
ChatGPT_safe_generate_response_OLD = Claude_safe_generate_response
GPT4_safe_generate_response = Claude_opus_safe_generate_response


# ============================================================================
# ###################[SECTION 6: TESTING] ####################################
# ============================================================================

if __name__ == "__main__":
    print("Testing Claude Agent SDK with ClaudeSDKClient (persistent connection)...")
    print(
        f"Compaction threshold: {COMPACTION_THRESHOLD*100:.0f}% ({COMPACTION_TOKEN_LIMIT:,} tokens)"
    )
    print()

    # Test multiple calls to see speedup
    prompts = [
        "Say exactly: Hello from test 1",
        "Say exactly: Hello from test 2",
        "Say exactly: Hello from test 3",
    ]

    times = []
    for i, prompt in enumerate(prompts):
        start = time.time()
        response = Claude_sonnet_request(prompt)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Call {i+1}: {elapsed:.2f}s - Response: {response[:50]}...")

    print()
    print(f"First call: {times[0]:.2f}s")
    print(f"Subsequent calls avg: {sum(times[1:])/len(times[1:]):.2f}s")
    print()

    # Show client stats
    stats = get_client_stats()
    print(f"Client pool stats: {json.dumps(stats, indent=2)}")

    # Cleanup
    cleanup_clients_sync()
    print("\nCleanup complete.")
