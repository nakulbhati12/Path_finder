import heapq
import time
from typing import List, Tuple, Dict, Optional, Set


class GridPathfinder:
    """Implements A* Search and Greedy Best-First Search on a 2D Grid."""

    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.goal = goal

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates Manhattan Distance heuristic h(n) between two grid cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns valid adjacent cells (Up, Down, Left, Right) that are not obstacles."""
        r, c = node
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def reconstruct_path(self, came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """Backtracks from Goal to Start to reconstruct the discovered path."""
        current = self.goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        return path

    def a_star_search(self) -> Tuple[Optional[List[Tuple[int, int]]], Set[Tuple[int, int]], float]:
        """
        Executes A* Search: f(n) = g(n) + h(n)
        Guarantees the shortest optimal path.
        """
        start_time = time.perf_counter()
        open_set = []
        heapq.heappush(open_set, (0, self.start))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {self.start: None}
        g_score: Dict[Tuple[int, int], int] = {self.start: 0}
        visited: Set[Tuple[int, int]] = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == self.goal:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return self.reconstruct_path(came_from), visited, elapsed_ms

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return None, visited, elapsed_ms

    def best_first_search(self) -> Tuple[Optional[List[Tuple[int, int]]], Set[Tuple[int, int]], float]:
        """
        Executes Greedy Best-First Search: f(n) = h(n)
        Expands the node closest to the goal based solely on heuristic.
        """
        start_time = time.perf_counter()
        open_set = []
        heapq.heappush(open_set, (self.heuristic(self.start, self.goal), self.start))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {self.start: None}
        visited: Set[Tuple[int, int]] = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == self.goal:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return self.reconstruct_path(came_from), visited, elapsed_ms

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor not in came_from:
                    came_from[neighbor] = current
                    h_score = self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (h_score, neighbor))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return None, visited, elapsed_ms

    def display_grid(self, path: Optional[List[Tuple[int, int]]], visited: Set[Tuple[int, int]]) -> None:
        """Renders the grid in the VS Code terminal with clear symbols."""
        path_set = set(path) if path else set()
        print("\n   " + " ".join(f"{c}" for c in range(self.cols)))
        print("  +" + "--" * self.cols + "+")
        for r in range(self.rows):
            row_str = f"{r} | "
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.start:
                    row_str += "S "
                elif pos == self.goal:
                    row_str += "G "
                elif self.grid[r][c] == 1:
                    row_str += "# "
                elif pos in path_set:
                    row_str += "* "
                elif pos in visited:
                    row_str += ". "
                else:
                    row_str += "0 "
            print(row_str + "|")
        print("  +" + "--" * self.cols + "+")
        print("Legend: S = Start | G = Goal | # = Wall | * = Path | . = Explored | 0 = Open\n")


def get_sample_grid() -> Tuple[List[List[int]], Tuple[int, int], Tuple[int, int]]:
    """Provides a 10x10 grid designed to highlight differences between A* and Best-First Search."""
    grid = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    ]
    return grid, (0, 0), (9, 9)


def main():
    print("=" * 65)
    print("   AI LAB MINI PROJECT: A* & BEST-FIRST SEARCH PATHFINDER")
    print("   Submitted by: YASHRAJ SINGH (243501217) & NAKUL BHATI (243501135)")
    print("=" * 65)

    grid, start, goal = get_sample_grid()
    pathfinder = GridPathfinder(grid, start, goal)

    while True:
        print("\nSelect an Option:")
        print("1. Run A* Search Algorithm (Optimal Path)")
        print("2. Run Best-First Search Algorithm (Greedy Heuristic)")
        print("3. Compare Both Algorithms Side-by-Side")
        print("4. Custom Start & Goal Coordinates")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            path, visited, elapsed = pathfinder.a_star_search()
            print("\n--- A* SEARCH RESULT ---")
            if path:
                print(f"Path Found! Steps: {len(path) - 1} | Nodes Explored: {len(visited)} | Time: {elapsed:.4f} ms")
                print(f"Coordinates: {path}")
                pathfinder.display_grid(path, visited)
            else:
                print("No valid path exists between Start and Goal.")

        elif choice == "2":
            path, visited, elapsed = pathfinder.best_first_search()
            print("\n--- BEST-FIRST SEARCH RESULT ---")
            if path:
                print(f"Path Found! Steps: {len(path) - 1} | Nodes Explored: {len(visited)} | Time: {elapsed:.4f} ms")
                print(f"Coordinates: {path}")
                pathfinder.display_grid(path, visited)
            else:
                print("No valid path exists between Start and Goal.")

        elif choice == "3":
            astar_path, astar_vis, astar_time = pathfinder.a_star_search()
            bfs_path, bfs_vis, bfs_time = pathfinder.best_first_search()

            print("\n" + "=" * 58)
            print(f"{'Metric':<25} | {'A* Search':<13} | {'Best-First':<13}")
            print("-" * 58)
            a_steps = len(astar_path) - 1 if astar_path else "N/A"
            b_steps = len(bfs_path) - 1 if bfs_path else "N/A"
            print(f"{'Path Length (Steps)':<25} | {str(a_steps):<13} | {str(b_steps):<13}")
            print(f"{'Nodes Explored':<25} | {len(astar_vis):<13} | {len(bfs_vis):<13}")
            print(f"{'Execution Time (ms)':<25} | {astar_time:<13.4f} | {bfs_time:<13.4f}")
            print(f"{'Optimality Guaranteed':<25} | {'Yes':<13} | {'No':<13}")
            print("=" * 58)

            print("\n[A* Grid Visualization]")
            pathfinder.display_grid(astar_path, astar_vis)
            print("[Best-First Search Grid Visualization]")
            pathfinder.display_grid(bfs_path, bfs_vis)

        elif choice == "4":
            try:
                sr = int(input("Enter Start Row (0-9): "))
                sc = int(input("Enter Start Col (0-9): "))
                gr = int(input("Enter Goal Row (0-9): "))
                gc = int(input("Enter Goal Col (0-9): "))
                if not (0 <= sr < 10 and 0 <= sc < 10 and 0 <= gr < 10 and 0 <= gc < 10):
                    print("Error: Coordinates must be between 0 and 9.")
                elif grid[sr][sc] == 1 or grid[gr][gc] == 1:
                    print("Error: Selected coordinate is blocked by an obstacle (#).")
                else:
                    pathfinder = GridPathfinder(grid, (sr, sc), (gr, gc))
                    print(f"Updated Start to {(sr, sc)} and Goal to {(gr, gc)}.")
            except ValueError:
                print("Invalid input. Please enter integer values.")

        elif choice == "5":
            print("Exiting AI Pathfinder Mini Project. Thank you!")
            break
        else:
            print("Invalid option. Please choose between 1 and 5.")


if __name__ == "__main__":
    main()
