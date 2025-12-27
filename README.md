# Claudeville: Generative Agents with Claude

Fork of Stanford's [Generative Agents](https://github.com/joonspk-research/generative_agents) ported from OpenAI API to Claude CLI for Max subscription users.

<p align="center" width="100%">
<img src="cover.png" alt="Smallville" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

## Current Status

**Minimal viable port.** The simulation runs using Claude CLI instead of OpenAI API.

### What Works

- Claude CLI integration (GPT API calls replaced with CLI subprocess calls)
- Basic simulation loop (perceive, plan, act, reflect)
- Memory storage (events, thoughts, chats in JSON)
- Keyword-based memory retrieval with recency/importance scoring
- Frontend visualization (Phaser.js)

### What Was Removed

- All embedding code (no more `text-embedding-ada-002` or cosine similarity)
- OpenAI dependency

## Requirements

- **Claude Code CLI** with Max subscription
- Python 3.9+
- Conda (recommended)

## Quick Start

```bash
git clone <repo>
cd claudeville
./start.sh
```

The script will:
1. Create conda environment if needed
2. Start Django frontend on http://localhost:8000
3. Start the backend simulation server

When prompted:
- Fork simulation: `base_the_ville_isabella_maria_klaus`
- New simulation name: anything you want

Then:
```
run <step-count>
```

Open http://localhost:8000 in browser to watch.

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
│   ├── reverie.py            # Main simulation loop
│   └── persona/
│       ├── cognitive_modules/
│       │   ├── perceive.py   # Environment perception
│       │   ├── retrieve.py   # Memory retrieval
│       │   ├── plan.py       # Action planning
│       │   ├── reflect.py    # Self-reflection
│       │   └── converse.py   # Conversations
│       └── prompt_template/
│           ├── claude_structure.py  # Claude CLI wrapper
│           └── run_prompt.py        # Prompt execution
└── environment/frontend_server/
    ├── storage/              # Simulation data
    └── templates/            # Phaser.js game
```

## Known Issues

- Simulation is slow even during sleep cycles
- Browser refresh may desync frontend/backend

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
