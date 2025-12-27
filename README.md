# Claudeville: Generative Agents with Claude

Fork of Stanford's [Generative Agents](https://github.com/joonspk-research/generative_agents) ported from OpenAI API to Claude Code CLI for Max subscription users.

<p align="center" width="100%">
<img src="cover.png" alt="Smallville" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

## Current Status

**Major architectural refactor complete.** This is no longer a minimal port - it's a fundamentally different approach to running generative agents.

### Key Departures from Original

| Original (Stanford) | Claudeville |
|---------------------|-------------|
| OpenAI API calls | Claude Agent SDK with persistent sessions |
| Multi-step cognitive chain | Single unified LLM call per step |
| File-based frontend polling | HTTP-based communication |
| Embedding-based retrieval | Keyword + recency scoring |
| Stateless API calls | Context window monitoring with compaction |

### What Works

- **Claude Agent SDK Integration**: Persistent connections with ~3x faster subsequent calls (~2.5s vs ~7-10s)
- **Unified Prompting System**: One LLM call per step returns action, social, and thought decisions
- **HTTP Backend/Frontend**: Flask server with `/movements`, `/status`, `/save` endpoints
- **Smart LLM Skip Logic**: Avoids redundant calls when actions are in progress
- **Parallel Persona Execution**: All personas run concurrently per simulation step
- **Memory storage**: Events, thoughts, chats in JSON with keyword-based retrieval

### What Was Removed

- All embedding code (no more `text-embedding-ada-002` or cosine similarity)
- OpenAI dependency
- Old prompt template directories (v1, v2, v3_ChatGPT)
- File-based polling (eliminated ~5000 JSON files per simulation)
- Multi-step cognitive chain (perceive, plan, execute, reflect as separate LLM calls)

## Requirements

- **Claude Max subscription** (for Claude Agent SDK access)
- Conda

## Quick Start

```bash
git clone <repo>
cd claudeville
./start.sh
```

The script will:
1. Create conda environment if needed
2. Start Django frontend on http://localhost:8000
3. Start Flask backend on http://localhost:5000
4. Start the CLI simulation controller

When prompted, press Enter for defaults or choose:
- `c` - Continue last simulation
- `custom` - Specify fork and simulation name
- Enter - Start new simulation with auto-generated name

Then in the CLI:
```
run <step-count>   # Run N simulation steps
status             # Show simulation info
personas           # Show all personas and their current actions
save               # Save simulation state
exit               # Save and quit
```

Open http://localhost:8000 in browser to watch the simulation animate.

## Manual Setup

### 1. Create Environment

```bash
conda env create -f environment.yaml
conda activate claudeville
```

### 2. Start Servers

Terminal 1 (Frontend):
```bash
cd environment/frontend_server
python manage.py runserver
```

Terminal 2 (Backend):
```bash
cd reverie/backend_server
python reverie.py
```

## Project Structure

```
claudeville/
├── start.sh                  # One-command startup
├── environment.yaml          # Conda environment
├── reverie/backend_server/
│   ├── reverie.py            # Main loop + Flask server
│   ├── cli_interface.py      # CLI commands
│   └── persona/
│       ├── persona.py        # Main persona class
│       ├── cognitive_modules/
│       │   └── perceive.py   # Environment perception
│       ├── memory_structures/
│       │   ├── spatial_memory.py
│       │   ├── associative_memory.py
│       │   └── scratch.py
│       └── prompt_template/
│           └── claude_structure.py  # UnifiedPersonaClient + SDK
└── environment/frontend_server/
    ├── storage/              # Simulation data
    └── templates/            # Phaser.js game
```

## Known Issues

- Avatar/sprite loading shows same character for all personas (frontend JS bug)

## Acknowledgements

This is a fork of [Generative Agents](https://github.com/joonspk-research/generative_agents) by Joon Sung Park et al. at Stanford. Please cite the original paper:

```bibtex
@inproceedings{Park2023GenerativeAgents,
  author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  title = {Generative Agents: Interactive Simulacra of Human Behavior},
  year = {2023},
  publisher = {Association for Computing Machinery},
  booktitle = {UIST '23}
}
```

Game assets:
- Background art: [PixyMoon](https://twitter.com/_PixyMoon_)
- Furniture/interior: [LimeZu](https://twitter.com/lime_px)
- Characters: [ぴぽ](https://twitter.com/pipohi)
