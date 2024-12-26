import networkx as nx
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import heapq

# A global set to keep track of visited cities
visited_cities = set()

def adjust_for_blocked_roads(road_connections, blocked_roads):
    """
    Adjusts the road network to account for blocked roads.

    Args:
        road_connections (dict): Dictionary mapping cities to their neighbors with distances.
        blocked_roads (list): List of tuples representing blocked roads (city1, city2).

    Returns:
        dict: Adjusted road connections with blocked roads removed.
    """
    adjusted_map = defaultdict(list)
    for city, neighbors in road_connections.items():
        for neighbor, distance in neighbors:
            if (city, neighbor) not in blocked_roads and (neighbor, city) not in blocked_roads:
                adjusted_map[city].append((neighbor, distance))
    return adjusted_map

def plot_road_network(city_list, road_connections, search_type=None, starting_city=None, destination_city=None):
    """
    Plots the road network using NetworkX with optional visualization for the path found.

    Args:
        city_list (list): List of city names.
        road_connections (dict): Dictionary mapping cities to their neighbors with distances.
        search_type (str, optional): Search method ('bfs', 'dfs', 'cost_bfs', 'traverse_all', or 'k_shortest').
        starting_city (str, optional): City to start the search.
        destination_city (str, optional): City to end the search.

    Returns:
        None
    """
    # Create an undirected graph
    network = nx.Graph()

    # Add edges with weights to the graph
    for city, neighbors in road_connections.items():
        for neighbor, distance in neighbors:
            network.add_edge(city, neighbor, weight=distance)

    # Position nodes in a circular layout
    positions = nx.circular_layout(network)
    plt.figure(figsize=(10, 7))

    # Determine node colors based on start and goal
    node_colors = []
    for node in network.nodes():
        if node == starting_city:
            node_colors.append('lime')  
        elif node == destination_city:
            node_colors.append('orange')  
        else:
            node_colors.append('cyan') 

    nx.draw(network, positions, node_color=node_colors, node_size=800, with_labels=True, font_weight='bold')
    edge_labels = nx.get_edge_attributes(network, 'weight')
    nx.draw_networkx_edge_labels(network, positions, edge_labels=edge_labels)

    # Highlight the path if a search is performed
    if search_type and starting_city and destination_city:
        if search_type == 'k_shortest':
            paths = k_shortest_paths(city_list, road_connections, starting_city, destination_city, k=3)
            for path, _ in paths:
                edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(network, positions, edgelist=edges, edge_color='purple', width=2)
        else:
            path, cost = find_path(city_list, road_connections, starting_city, destination_city, search_type)
            if path:
                edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(network, positions, edgelist=edges, edge_color='red', width=2)
                plt.title(f"{search_type.upper()} Path - Total Distance: {cost} km")
            else:
                plt.title(f"No path found using {search_type.upper()}")
    else:
        plt.title("Road Network")

    plt.show()

def find_path(city_list, road_connections, starting_city, destination_city, search_type):
    """
    Finds a path using the specified search method.

    Args:
        city_list (list): List of city names.
        road_connections (dict): Dictionary mapping cities to neighbors with distances.
        starting_city (str): Starting city.
        destination_city (str): Goal city.
        search_type (str): Search strategy ('bfs', 'dfs', 'cost_bfs', 'traverse_all').

    Returns:
        tuple: (path as a list, total distance as an integer)
    """
    visited_cities.clear()

    if search_type == 'bfs':
        return breadth_first_search(city_list, road_connections, starting_city, destination_city)
    elif search_type == 'dfs':
        return depth_first_search(city_list, road_connections, starting_city, destination_city)
    elif search_type == 'cost_bfs':
        return cost_sensitive_bfs(city_list, road_connections, starting_city, destination_city)
    elif search_type == 'traverse_all':
        return traverse_all_cities(city_list, road_connections, starting_city)
    else:
        raise ValueError("Unsupported search type. Use 'bfs', 'dfs', 'cost_bfs', or 'traverse_all'.")

def breadth_first_search(city_list, road_connections, starting_city, destination_city):
    """Implements BFS to find the shortest unweighted path."""
    if starting_city == destination_city:
        return ([starting_city], 0)

    queue = deque([starting_city])
    paths = {starting_city: (None, 0)}

    while queue:
        current_city = queue.popleft()
        visited_cities.add(current_city)

        if current_city == destination_city:
            break

        for neighbor, cost in road_connections[current_city]:
            if neighbor not in visited_cities and neighbor not in paths:
                queue.append(neighbor)
                paths[neighbor] = (current_city, 1)

    if destination_city not in paths:
        return None, 0

    path = []
    current = destination_city
    total_cost = 0

    while current:
        path.append(current)
        prev, step_cost = paths[current]
        total_cost += step_cost
        current = prev

    return path[::-1], total_cost

def depth_first_search(city_list, road_connections, starting_city, destination_city):
    """Implements DFS to find a feasible path."""
    visited_cities.add(starting_city)
    current_path = [starting_city]

    if starting_city == destination_city:
        return current_path, 0

    for neighbor, cost in road_connections[starting_city]:
        if neighbor not in visited_cities:
            path, path_cost = depth_first_search(city_list, road_connections, neighbor, destination_city)
            if path:
                return current_path + path, path_cost + cost

    return None, 0

def cost_sensitive_bfs(city_list, road_connections, starting_city, destination_city):
    """Implements BFS considering path costs."""
    if starting_city == destination_city:
        return ([starting_city], 0)

    queue = deque([starting_city])
    paths = {starting_city: (None, 0)}

    while queue:
        current_city = queue.popleft()
        visited_cities.add(current_city)
        current_cost = paths[current_city][1]

        for neighbor, cost in road_connections[current_city]:
            new_cost = current_cost + cost
            if neighbor not in paths or new_cost < paths[neighbor][1]:
                paths[neighbor] = (current_city, new_cost)
                if neighbor not in visited_cities:
                    queue.append(neighbor)

    if destination_city not in paths:
        return None, 0

    path = []
    current = destination_city

    while current:
        path.append(current)
        current = paths[current][0]

    return path[::-1], paths[destination_city][1]

def traverse_all_cities(city_list, road_connections, starting_city):
    """Traverses all cities starting from a given city."""
    def dfs_traverse(city, visited, path, cost):
        if len(visited) == len(city_list):
            return path, cost

        best_path, best_cost = None, float('inf')
        for neighbor, distance in road_connections[city]:
            if neighbor not in visited:
                new_path, new_cost = dfs_traverse(neighbor, visited | {neighbor}, path + [neighbor], cost + distance)
                if new_cost < best_cost:
                    best_path, best_cost = new_path, new_cost
        return best_path, best_cost

    return dfs_traverse(starting_city, {starting_city}, [starting_city], 0)

def k_shortest_paths(city_list, road_connections, starting_city, destination_city, k):
    """Finds k-shortest paths between two cities."""
    def path_cost(path):
        return sum(
            road_connections[path[i]][j][1] for i, j in enumerate(range(1, len(path)))
        )

    priority_queue = [(0, [starting_city])]
    shortest_paths = []

    while priority_queue and len(shortest_paths) < k:
        cost, path = heapq.heappop(priority_queue)
        current = path[-1]

        if current == destination_city:
            shortest_paths.append((path, cost))

        for neighbor, distance in road_connections[current]:
            if neighbor not in path:
                heapq.heappush(priority_queue, (cost + distance, path + [neighbor]))

    return shortest_paths

# Cities and roads definition
ethiopian_cities = ['Addis Ababa', 'Bahir Dar', 'Gondar', 'Hawassa', 'Mekelle', 'Dire Dawa', 'Jimma', 'Dessie', 'Jijiga']
road_map = {
    'Addis Ababa': [('Hawassa', 275), ('Bahir Dar', 510), ('Dire Dawa', 450), ('Jimma', 350)],
    'Bahir Dar': [('Addis Ababa', 510), ('Gondar', 180), ('Dessie', 300)],
    'Gondar': [('Bahir Dar', 180), ('Mekelle', 300), ('Dessie', 360)],
    'Hawassa': [('Mekelle', 1000), ('Addis Ababa', 275), ('Dire Dawa', 600)],
    'Mekelle': [('Gondar', 300), ('Hawassa', 1000), ('Dessie', 400)],
    'Dire Dawa': [('Addis Ababa', 450), ('Hawassa', 600), ('Jijiga', 150)],
    'Jimma': [('Addis Ababa', 350), ('Hawassa', 400)],
    'Dessie': [('Bahir Dar', 300), ('Gondar', 360), ('Mekelle', 400)],
    'Jijiga': [('Dire Dawa', 150)]
}

def main():
    """Demonstrates the functionality with predefined inputs."""
    print("Starting BFS example...")
    path, cost = find_path(ethiopian_cities, road_map, 'Addis Ababa', 'Mekelle', 'bfs')
    print(f"BFS Path: {path} with cost {cost} km")

    print("Starting DFS example...")
    path, cost = find_path(ethiopian_cities, road_map, 'Addis Ababa', 'Mekelle', 'dfs')
    print(f"DFS Path: {path} with cost {cost} km")

    print("Starting Cost-Sensitive BFS example...")
    path, cost = find_path(ethiopian_cities, road_map, 'Addis Ababa', 'Mekelle', 'cost_bfs')
    print(f"Cost-Sensitive BFS Path: {path} with cost {cost} km")

    print("Traversing all cities starting from Addis Ababa...")
    path, cost = find_path(ethiopian_cities, road_map, 'Addis Ababa', None, 'traverse_all')
    print(f"Traversal Path: {path} with cost {cost} km")

    print("Finding K-Shortest Paths example...")
    k_paths = k_shortest_paths(ethiopian_cities, road_map, 'Addis Ababa', 'Mekelle', 3)
    for i, (path, cost) in enumerate(k_paths):
        print(f"Path {i + 1}: {path} with cost {cost} km")

    print("Handling dynamic road conditions (blocking Addis Ababa to Bahir Dar)...")
    blocked = [('Addis Ababa', 'Bahir Dar')]
    adjusted_map = adjust_for_blocked_roads(road_map, blocked)
    path, cost = find_path(ethiopian_cities, adjusted_map, 'Addis Ababa', 'Mekelle', 'bfs')
    print(f"Adjusted BFS Path: {path} with cost {cost} km")

    # Visualization example
    plot_road_network(ethiopian_cities, adjusted_map, 'bfs', 'Addis Ababa', 'Mekelle')

if __name__ == "__main__":
    main()
