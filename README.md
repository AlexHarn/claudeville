# Claudeville: Generative Agents with Claude

> **WORK IN PROGRESS FORK**
>
> This is an experimental fork of Stanford's [Generative Agents](https://github.com/joonspk-research/generative_agents) project,
> porting from OpenAI API to Claude CLI for Max subscription users.

<p align="center" width="100%">
<img src="cover.png" alt="Smallville" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

## Current Status

**This is a minimal viable port.** The simulation runs using Claude CLI instead of OpenAI API.

### What Works Now

- **Claude CLI Integration**: GPT API calls replaced with Claude CLI subprocess calls
- **Basic Simulation Loop**: Personas perceive, plan, act, and reflect
- **Memory Storage**: Events, thoughts, and chats stored in JSON files
- **Keyword-Based Retrieval**: Memory retrieval uses keyword matching and recency/importance scoring
- **Frontend Visualization**: Phaser.js game displays personas moving and acting

### What Was Removed

- **All Embedding Code**: No more `text-embedding-ada-002` calls or cosine similarity
- **OpenAI Dependency**: Completely removed from requirements

### What's NOT Implemented Yet (Planned)

The README previously described an ambitious architecture that is NOT yet built:

- ❌ **Persistent Claude Sessions**: Currently each prompt is a fresh CLI call (like original)
- ❌ **Subconscious Retriever**: Planned Sonnet-powered semantic memory search
- ❌ **Emotional State Tracking**: Not yet added to scratch.py
- ❌ **Context Compaction**: No token tracking or manual compaction yet
- ❌ **Per-Persona Model Config**: All personas use same model

The current implementation is essentially a **hot-swap of GPT → Claude CLI** with embeddings gutted.

## Requirements

- **Claude Code CLI** with Max subscription (no API key needed)
- Python 3.9+
- Conda/Mamba (recommended)

## Quick Start

```bash
# Clone and enter directory
git clone <repo>
cd claudeville

# Run the startup script
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

### Step 1. Create Utils File

In `reverie/backend_server`, create `utils.py`:
```python
# Claude CLI Configuration (no API key needed)
key_owner = "<Your Name>"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"
debug = True
```

### Step 2. Install Dependencies

Using conda:
```bash
conda env create -f environment.yaml
conda activate claudeville
```

Or pip:
```bash
pip install -r requirements.txt
```

### Step 3. Start Servers

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
├── CLAUDE.md                 # Project status & notes
├── reverie/backend_server/
│   ├── reverie.py            # Main simulation loop
│   └── persona/
│       ├── persona.py        # Persona class
│       ├── cognitive_modules/
│       │   ├── perceive.py   # Environment perception
│       │   ├── retrieve.py   # Memory retrieval (keyword-based)
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

- **Performance**: Simulation is slow even during sleep cycles (no LLM calls)
- **Browser Refresh**: May need to restart backend if frontend loses sync
- **Verbose Output**: Some debug prints may still appear

## Future Plans

See `.claude/` directory and `CLAUDE.md` for detailed architectural plans including:
- Persistent Claude sessions per persona
- Subconscious Sonnet-powered memory retrieval
- Emotional state tracking
- Parallel persona execution

## Original Authors and Citation

**Original Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein

Please cite the original paper:
```bibtex
@inproceedings{Park2023GenerativeAgents,
  author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  title = {Generative Agents: Interactive Simulacra of Human Behavior},
  year = {2023},
  publisher = {Association for Computing Machinery},
  booktitle = {UIST '23},
  location = {San Francisco, CA, USA}
}
```

## Acknowledgements

Original game assets by:
* Background art: [PixyMoon (@_PixyMoon\_)](https://twitter.com/_PixyMoon_)
* Furniture/interior design: [LimeZu (@lime_px)](https://twitter.com/lime_px)
* Character design: [ぴぽ (@pipohi)](https://twitter.com/pipohi)

This fork by Alex, using Claude CLI with Anthropic's Max subscription.
