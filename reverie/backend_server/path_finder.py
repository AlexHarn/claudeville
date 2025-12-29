"""
Original Author: Joon Sung Park (joonspk@stanford.edu)
Heavily modified for Claudeville (Claude CLI port)

File: path_finder.py
Description: Implements PathFinder class for generative agents path finding.
"""

from typing import Optional

import numpy as np


class PathFinder:
    """
    A path finding utility class for navigating maze environments.

    Uses a breadth-first wave propagation algorithm to find shortest paths
    between points while avoiding collision blocks.
    """

    def __init__(self, maze: list, collision_block_id: str, extra_blocked: set = None):
        """
        Initialize PathFinder with a maze and collision block identifier.

        Args:
            maze: 2D list representing the maze grid
            collision_block_id: Character/value identifying impassable blocks
            extra_blocked: Optional set of (x, y) tiles to treat as blocked
        """
        self.maze = maze
        self.collision_block_id = collision_block_id
        self.extra_blocked = extra_blocked or set()

    def find_path(self, start: tuple, end: tuple) -> list:
        """
        Find the shortest path from start to end coordinates.

        Uses wave propagation (BFS) algorithm for shortest path.
        Coordinates are in (x, y) format and internally converted.

        Args:
            start: Starting coordinate as (x, y) tuple
            end: Ending coordinate as (x, y) tuple

        Returns:
            List of (x, y) coordinate tuples forming the path from start to end.
            Returns path with just the start if no path exists.
        """
        # Convert from (x, y) to internal (row, col) format
        internal_start = (start[1], start[0])
        internal_end = (end[1], end[0])

        path = self._find_path_internal(internal_start, internal_end)

        # Convert back to (x, y) format
        return [(coord[1], coord[0]) for coord in path]

    def find_path_to_nearest(self, start: tuple, targets: list) -> tuple:
        """
        Find the path to the nearest target from a list of targets.

        Args:
            start: Starting coordinate as (x, y) tuple
            targets: List of potential target coordinates as (x, y) tuples

        Returns:
            Tuple of (path, target_reached) where path is the list of coordinates
            and target_reached is the coordinate that was reached.
            Returns ([], None) if no valid targets.
        """
        if not targets:
            return ([], None)

        closest_target = None
        shortest_path = None

        for target in targets:
            path = self.find_path(start, target)
            if shortest_path is None or len(path) < len(shortest_path):
                shortest_path = path
                closest_target = target

        return (shortest_path, closest_target)

    def _is_valid_position(self, row: int, col: int) -> bool:
        """
        Check if a position is valid (within bounds and not a collision block).

        Args:
            row: Row index in the maze
            col: Column index in the maze

        Returns:
            True if position is valid for traversal, False otherwise
        """
        if row < 0 or col < 0:
            return False
        if row >= len(self.maze) or col >= len(self.maze[0]):
            return False
        return self.maze[row][col] != self.collision_block_id

    def _find_path_internal(self, start: tuple, end: tuple) -> list:
        """
        Internal path finding using wave propagation algorithm.

        Args:
            start: Starting coordinate as (row, col) tuple
            end: Ending coordinate as (row, col) tuple

        Returns:
            List of (row, col) coordinate tuples forming the path
        """
        # Build collision map (1 = blocked, 0 = passable)
        collision_map = []
        for row_idx, row in enumerate(self.maze):
            new_row = []
            for col_idx, cell in enumerate(row):
                # Check base collision and extra blocked tiles
                # Note: extra_blocked uses (x, y) format, maze uses (row, col) = (y, x)
                if (
                    cell == self.collision_block_id
                    or (col_idx, row_idx) in self.extra_blocked
                ):
                    new_row.append(1)
                else:
                    new_row.append(0)
            collision_map.append(new_row)

        # Initialize distance map
        distance_map = []
        for i in range(len(collision_map)):
            distance_map.append([0] * len(collision_map[i]))

        # Set starting position
        start_row, start_col = start
        distance_map[start_row][start_col] = 1

        # Wave propagation with iteration limit
        step = 0
        max_iterations = 150
        while distance_map[end[0]][end[1]] == 0 and step < max_iterations:
            step += 1
            self._propagate_wave(collision_map, distance_map, step)

        # Trace path back from end to start
        row, col = end
        current_distance = distance_map[row][col]
        path = [(row, col)]

        while current_distance > 1:
            # Check all four directions for the previous step
            if row > 0 and distance_map[row - 1][col] == current_distance - 1:
                row = row - 1
                path.append((row, col))
                current_distance -= 1
            elif col > 0 and distance_map[row][col - 1] == current_distance - 1:
                col = col - 1
                path.append((row, col))
                current_distance -= 1
            elif (
                row < len(distance_map) - 1
                and distance_map[row + 1][col] == current_distance - 1
            ):
                row = row + 1
                path.append((row, col))
                current_distance -= 1
            elif (
                col < len(distance_map[row]) - 1
                and distance_map[row][col + 1] == current_distance - 1
            ):
                col = col + 1
                path.append((row, col))
                current_distance -= 1
            else:
                # No valid path found
                break

        path.reverse()
        return path

    def _propagate_wave(
        self, collision_map: list, distance_map: list, step: int
    ) -> None:
        """
        Propagate the distance wave one step outward.

        Args:
            collision_map: 2D list where 1 = blocked, 0 = passable
            distance_map: 2D list of distances from start
            step: Current step number in wave propagation
        """
        for i in range(len(distance_map)):
            for j in range(len(distance_map[i])):
                if distance_map[i][j] == step:
                    # Propagate to adjacent unvisited, passable cells
                    if (
                        i > 0
                        and distance_map[i - 1][j] == 0
                        and collision_map[i - 1][j] == 0
                    ):
                        distance_map[i - 1][j] = step + 1
                    if (
                        j > 0
                        and distance_map[i][j - 1] == 0
                        and collision_map[i][j - 1] == 0
                    ):
                        distance_map[i][j - 1] = step + 1
                    if (
                        i < len(distance_map) - 1
                        and distance_map[i + 1][j] == 0
                        and collision_map[i + 1][j] == 0
                    ):
                        distance_map[i + 1][j] = step + 1
                    if (
                        j < len(distance_map[i]) - 1
                        and distance_map[i][j + 1] == 0
                        and collision_map[i][j + 1] == 0
                    ):
                        distance_map[i][j + 1] = step + 1

    @staticmethod
    def closest_coordinate(curr: tuple, target_list: list) -> Optional[tuple]:
        """
        Find the closest coordinate from a list of targets using Euclidean distance.

        Args:
            curr: Current coordinate as (x, y) tuple
            target_list: List of target coordinates as (x, y) tuples

        Returns:
            The closest coordinate tuple, or None if target_list is empty
        """
        if not target_list:
            return None

        min_dist = None
        closest = None

        for coordinate in target_list:
            a = np.array(coordinate)
            b = np.array(curr)
            dist = float(np.linalg.norm(a - b))
            if closest is None or dist < min_dist:
                min_dist = dist
                closest = coordinate

        return closest


# Backwards compatibility wrapper for existing code
def path_finder(maze, start, end, collision_block_char, verbose=False):
    """
    Legacy wrapper function for backwards compatibility.

    Args:
        maze: 2D list representing the maze grid
        start: Starting coordinate as (x, y) tuple
        end: Ending coordinate as (x, y) tuple
        collision_block_char: Character identifying impassable blocks
        verbose: Unused, kept for API compatibility

    Returns:
        List of (x, y) coordinate tuples forming the path
    """
    pf = PathFinder(maze, collision_block_char)
    return pf.find_path(start, end)


def closest_coordinate(curr_coordinate, target_coordinates):
    """
    Legacy wrapper function for backwards compatibility.

    Args:
        curr_coordinate: Current coordinate as (x, y) tuple
        target_coordinates: List of target coordinates as (x, y) tuples

    Returns:
        The closest coordinate tuple, or None if empty
    """
    return PathFinder.closest_coordinate(curr_coordinate, target_coordinates)


if __name__ == "__main__":
    maze = [
        ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
        [" ", " ", "#", " ", " ", " ", " ", " ", "#", " ", " ", " ", "#"],
        ["#", " ", "#", " ", " ", "#", "#", " ", " ", " ", "#", " ", "#"],
        ["#", " ", "#", " ", " ", "#", "#", " ", "#", " ", "#", " ", "#"],
        ["#", " ", " ", " ", " ", " ", " ", " ", "#", " ", " ", " ", "#"],
        ["#", "#", "#", " ", "#", " ", "#", "#", "#", " ", "#", " ", "#"],
        ["#", " ", " ", " ", " ", " ", " ", " ", " ", " ", "#", " ", " "],
        ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
    ]

    # Test PathFinder class
    pf = PathFinder(maze, "#")

    print("Test 1: Same start and end")
    start = (0, 1)
    end = (0, 1)
    print(pf.find_path(start, end))

    print("\nTest 2: Path finding")
    start = (0, 1)
    end = (11, 4)
    print(pf.find_path(start, end))

    print("\nTest 3: Find path to nearest")
    start = (0, 1)
    targets = [(11, 4), (12, 6), (5, 3)]
    path, target = pf.find_path_to_nearest(start, targets)
    print(f"Path: {path}")
    print(f"Target reached: {target}")

    print("\nTest 4: Closest coordinate")
    curr = (0, 1)
    coords = [(11, 4), (12, 6), (5, 3)]
    print(PathFinder.closest_coordinate(curr, coords))

    print("\nTest 5: Legacy wrapper compatibility")
    print(path_finder(maze, (0, 1), (11, 4), "#"))
