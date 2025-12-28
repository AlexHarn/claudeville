# Claudeville: Generative Agents with Claude

Fork of Stanford's [Generative Agents](https://github.com/joonspk-research/generative_agents) ported from OpenAI API to Claude Agent SDK for Max subscription users.

## Current Status

**Full-screen game interface with frontend-driven simulation.** This is no longer a minimal port - it's a fundamentally different approach to running generative agents with a modern, game-like UI.

### Key Departures from Original

| Original (Stanford) | Claudeville |
|---------------------|-------------|
| OpenAI API calls | Claude Agent SDK with persistent sessions |
| Multi-step cognitive chain | Single unified LLM call per step |
| File-based frontend polling | HTTP-based communication |
| Embedding-based retrieval | Keyword + recency scoring |
| Stateless API calls | Context window monitoring with compaction |
| Fixed canvas, page layout | Full-screen game with floating UI panels |
| Backend-driven simulation | Frontend-driven with smart buffering |

## Features

### Game Interface

- **Full-screen Phaser.js game** that fills the browser window and resizes dynamically
- **Mouse controls**: Click and drag to pan the map, scroll wheel to zoom (0.3x - 3x)
- **Floating UI panels** with dark translucent styling:
  - Top control bar: time display, play/pause/skip, speed slider, zoom indicator
  - Right persona panel: collapsible list of all personas with live action/location
  - Bottom chat popup: shows active conversations with speech bubbles
- **Click persona cards** to center the camera on that character
- **ESC menu**: Save game, view saved simulations

### Conversations

- **Group conversations**: 3+ personas can naturally join ongoing conversations when nearby
- **Chat popup panel**: Bottom-center floating panel with:
  - Tabs for multiple simultaneous conversations (initials + "+N" for groups)
  - Click tab to pan camera to that conversation's location
  - Auto-selects closest conversation to camera
  - Minimizable with smooth animations
  - Auto-fades 5 seconds after conversation ends
- **Conversation positioning**: LLM prompted to stay stationary during dialogue, move closer for intimate chats
- **Shared conversation groups**: Backend tracks `ConversationGroup` objects, syncs chat lines across all participants

### Simulation Control

- **Frontend-driven simulation**: No need to use CLI commands - the frontend automatically requests simulation steps
- **Smart buffering**: Keeps 2-3 steps simulated ahead for smooth playback
- **Play/Pause**: Control animation playback without stopping simulation
- **Skip to Next Action**: Fast-forward through walking/sleeping until an LLM decision happens
- **Speed control**: 1x to 10x playback speed

### Backend Architecture

- **Claude Agent SDK Integration**: Persistent connections with ~3x faster subsequent calls (~2.5s vs ~7-10s)
- **Unified Prompting System**: One LLM call per step returns action, social, and thought decisions
- **HTTP Backend/Frontend**: Flask server with `/movements`, `/status`, `/save`, `/simulate`, `/saves` endpoints
- **Smart LLM Skip Logic**: Avoids redundant calls when actions are in progress
- **Parallel Persona Execution**: All personas run concurrently per simulation step
- **Memory storage**: Events, thoughts, chats in JSON with keyword-based retrieval
- **Sleep-triggered compaction**: Context automatically compacted when persona goes to sleep
- **Conversation memory**: Recent conversations (with dialogue) included in model context on init/compaction

### What Was Removed

- All embedding code (no more `text-embedding-ada-002` or cosine similarity)
- OpenAI dependency
- Old prompt template directories (v1, v2, v3_ChatGPT)
- File-based polling (eliminated ~5000 JSON files per simulation)
- Multi-step cognitive chain (perceive, plan, execute, reflect as separate LLM calls)
- Bootstrap CSS framework (replaced with custom minimal CSS)
- Arrow key camera controls (replaced with mouse drag)

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
4. Open CLI for simulation management

When prompted, press Enter for defaults or choose:
- `c` - Continue last simulation
- `custom` - Specify fork and simulation name
- Enter - Start new simulation with auto-generated name

Open http://localhost:8000/simulator_home in browser - simulation runs automatically!

### Controls

| Control | Action |
|---------|--------|
| **Mouse drag** | Pan the map |
| **Scroll wheel** | Zoom in/out |
| **Play button** | Resume animation |
| **Pause button** | Pause animation (simulation continues buffering) |
| **Skip button** | Fast-forward to next LLM decision |
| **Speed slider** | Adjust playback speed (1x-10x) |
| **ESC** | Open menu (save game, view saves) |
| **Click persona card** | Center camera on that character |

### CLI Commands (Optional)

The CLI is still available for manual control:
```
run <N>    # Run N simulation steps manually
status     # Show simulation info
save       # Save simulation state
fin        # Save and exit
quit       # Exit without saving
```

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
    ├── static_dirs/css/      # Game UI styles
    ├── storage/              # Simulation data
    └── templates/home/       # Phaser.js game + UI
```

## Known Issues

- Avatar/sprite loading shows same character for all personas (frontend JS bug)
- Persona panel needs improvements for large groups (25+ personas) - scrolling works but needs smart ordering by proximity/activity

## Roadmap

- [x] Conversation display (chat popup panel with speech bubbles)
- [ ] Smart persona panel ordering (by proximity to camera, recent actions, conversations)
- [ ] Persona panel search/filter for large simulations
- [ ] Minimap showing persona locations
- [ ] Click-to-follow: lock camera to follow a specific persona
- [ ] Lecture scenario support (one speaker, multiple listeners)

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
