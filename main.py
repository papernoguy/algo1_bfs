import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
from collections import deque
from networkx.algorithms import bipartite


# gets networkx graph  and a tuple of roots
# returns paths[vertex] = (vertex,...,root)
def bfs_multiple_roots(graph, roots):
    # Initialize: arr_queue[] representing an array of queues for every root. each queue is a tuple of vertex-name
    # and a path list to its closest root. Visited queue, and paths array.
    visited, arr_queue, paths = [None], [None] * len(roots), [None] * (len(graph) + 1)
    for i in range(0, len(roots)):
        arr_queue[i] = [(roots[i], [roots[i]])]
        visited.append(roots[i])

    # BFS search from multiple roots at the same time
    while arr_queue[0]:
        for i in range(0, len(roots)):
            if arr_queue[i]:
                vertex, paths[vertex] = arr_queue[i].pop(0)
                print("node:", vertex, "path", paths[vertex], end="\n")  # BFS output
                for neighbour in graph[vertex]:
                    if neighbour not in visited:
                        visited.append(neighbour)
                        arr_queue[i].append((neighbour, [neighbour, ] + paths[vertex]))
    return paths


# Exercise 3 ###################################################################
# returns "even-odd graph" made of two nodes, even + odd nodes, for any given graph node,
# and two edges, even-to-odd and odd-to-even for any given graph edge
def eo_graph(graph):
    eo = nx.Graph()
    edges = list(graph.edges())
    for i in range(0, graph.number_of_nodes()):
        eo.add_nodes_from([i])
        eo.add_nodes_from([len(graph) + i])
    for i in range(0, graph.number_of_edges()):
        eo.add_edges_from([(edges[i][0], edges[i][1] + len(graph)), (edges[i][0] + len(graph), edges[i][1])])
    return eo


# returns shortest *even* paths for each vertex to its root (paths[vertex] = vertex,5,4,,,root)
def bfs_even_paths(graph, root):
    eog = eo_graph(graph)  # change input to even-odd graph
    visited = [root]
    queue = [(root, [root])]  # tuple of vertex-name and a path list to its closest root.
    paths = [None] * (len(eog) + 1)
    while queue:
        vertex, paths[vertex] = queue.pop(0)
        print("node:", vertex, "path", paths[vertex], end="\n")  # BFS output
        for neighbour in eog[vertex]:
            if neighbour not in visited:
                visited.append(neighbour)
                temp = neighbour
                if neighbour > len(graph):  # translate vertex name from even-odd graph back to the original
                    temp = neighbour - len(graph)
                queue.append((neighbour, [temp] + paths[vertex]))

    for i in range(0, len(graph)):  # delete all odd paths
        paths.pop()
    return paths


# Gets graph and a root
# Returns a list[vertex] of (distance-to root, pi-previous node)
# if vertex cannot connect to root list[vertex] = None
def bfs(graph, root):
    distance = 0
    pi = None
    visited = [root]
    queue = [(root, distance, pi)]  # Initialize a queue of tuples
    data = [None] * (len(graph) + 1)
    while queue:
        vertex, distance, pi = queue.pop(0)
        # print("node:", vertex, "distance", distance, end="\n")  # REGULAR BFS OUTPUT
        data[vertex] = (distance, pi)
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append((neighbour, distance + 1, vertex))

    print(data)
    return data


def recursiveBFS(graph, queue, visited, data):
    if not queue:
        return
    vertex, distance, pi = queue.pop(0)
    print("node:", vertex, "distance", distance, "pi", pi, end="\n")  # REGULAR BFS OUTPUT
    for neighbour in graph[vertex]:
        if neighbour not in visited:
            visited.append(neighbour)
            queue.append((neighbour, distance + 1, vertex))
    recursiveBFS(graph, queue, visited, data)


def is_minimum_degree2(graph, root):
    visited, queue, min_degree = [], [], []
    visited.append(root), queue.append(root)
    while queue:
        vertex = queue.pop()
        degree = len(graph[vertex])
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
        if len(min_degree) == 2:
            if degree < min_degree[1]:
                min_degree[1] = degree
            else:
                if degree < min_degree[0]:
                    min_degree[0] = degree
        else:
            min_degree.append(degree)
    if len(min_degree) == 2 and min_degree[0] + min_degree[1] > len(visited):
        print("is_minimum_degree2::: true")
        return True
    print("is_minimum_degree2::: false")
    return False


def sink_locator(matrix):
    i, j = 0, 0
    while i < len(matrix) and j < len(matrix):
        if matrix[i][j] == 0:
            j += 1
        else:
            i = j
    for k in range(0, len(matrix)):
        if matrix[k][i] == 0 and k != i:
            print("sink_locator::: sink not found")
            return None
    print("sink_locator:::", i, "is sink")
    return i

    # A recursive function that uses
    # visited[] and parent to detect
    # cycle in subgraph reachable from vertex v.

def dfs_cyclic(graph):
    return True


if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    g.add_edges_from([(1, 2), (1, 4), (2, 3), (3, 4), (5, 6), (1, 5), (7, 3), (6, 8), (7, 10), (8, 9), (9, 10)])
    # exercise2 = bfs_multiple_roots(g, (4, 6))
    # exercise3 = bfs_even_paths(g, 4)
    visited = []

    exercise4 = dfs_cyclic(g)
    # nx.draw(g3)
    # sub_ax1 = plt.subplot(121)
    # nx.draw(g3, with_labels=True, font_weight='bold')
    # plt.show()
