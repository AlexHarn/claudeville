# Claudeville Utils Package
# Configuration and utility functions

# Asset paths
maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

# Storage paths
fs_storage_base = (
    "../../environment/frontend_server/storage/base"  # Base templates (tracked in git)
)
fs_storage_runs = (
    "../../environment/frontend_server/storage/runs"  # Simulation runs (gitignored)
)
fs_temp_storage = "../../environment/frontend_server/temp_storage"

# Legacy alias for compatibility
fs_storage = fs_storage_runs

# Simulation settings
collision_block_id = "32125"  # Tile ID for collision detection in maze pathfinding
debug = True

# Re-export CSV utility for convenience
