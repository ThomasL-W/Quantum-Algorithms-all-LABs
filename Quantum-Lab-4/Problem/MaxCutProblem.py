import random
import networkx as nx
from sympy import symbols, Integer
from itertools import product


class MaxCutProblem:
    """
    Represents a Max-Cut optimization problem on a graph.

    The cost function to MINIMIZE is: - sum_{(vi,vj) in E} J_ij * s_vi * s_vj
    where s_v in {-1, +1} and J_ij is the edge weight (1 for unweighted graphs).
    """

    def __init__(self, graph: nx.Graph):
        """
        :param graph: A networkx Graph instance (weighted or unweighted).
        """
        self.graph = graph
        self._sympy_expr = None

    def to_sympy_expr(self):
        """
        Converts the Max-Cut graph to a sympy symbolic cost expression.
        Cached after the first computation.
        """
        if self._sympy_expr is not None:
            return self._sympy_expr

        from sympy import Integer
        expr = Integer(0)

        for u, v, data in self.graph.edges(data=True):
            weight = data.get("weight", 1)
            s_u = symbols(f"s_{u}")
            s_v = symbols(f"s_{v}")
            expr += -weight * s_u * s_v

        self._sympy_expr = expr
        return self._sympy_expr

    def eval(self, solution: dict) -> float:
        """
        Evaluates the cost function for a given solution.

        :param solution: Dict mapping variable name strings to {-1, +1}.
        :return: Numerical cost as a float.
        """
        expr = self.to_sympy_expr()
        subs_dict = {symbols(k): v for k, v in solution.items()}
        return float(expr.subs(subs_dict))

    def generate_complete_search_space(self):
        """Generates all {-1, +1}^n assignments for Max-Cut."""
        nodes = list(self.graph.nodes())
        return [
            {f"s_{nodes[i]}": combo[i] for i in range(len(nodes))}
            for combo in product([-1, 1], repeat=len(nodes))
        ]

    def generate_random_solution(self):
        """Generates a random {-1, +1} assignment for all nodes."""
        nodes = list(self.graph.nodes())
        return {f"s_{node}": random.choice([-1, 1]) for node in nodes}

    def gen_neighbor_sol(self, solution):
        """Generates a neighbor by flipping one random spin."""
        neighbor = dict(solution)
        nodes = list(self.graph.nodes())
        flip_node = random.choice(nodes)
        key = f"s_{flip_node}"
        neighbor[key] = -neighbor[key]
        return neighbor

    def display_graph(self):
        """Displays the graph with node and edge labels."""
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph, seed=42)
        labels = {node: f"$s_{{{node}}}$" for node in self.graph.nodes()}
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        plt.figure(figsize=(6, 5))
        nx.draw(self.graph, pos, labels=labels, with_labels=True,
                node_color="black", font_color="white", node_size=800, font_size=12)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Max-Cut Problem Graph")
        plt.tight_layout()
        plt.show()

    def display_solution(self, solution: dict):
        """
        Displays the graph with nodes colored by partition.
        Red: s=+1, Blue: s=-1.
        """
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph, seed=42)
        node_colors = [
            "red" if solution.get(f"s_{node}", 1) == 1 else "blue"
            for node in self.graph.nodes()
        ]
        labels = {node: f"$s_{{{node}}}$" for node in self.graph.nodes()}
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        cost = self.eval(solution)
        plt.figure(figsize=(6, 5))
        nx.draw(self.graph, pos, labels=labels, with_labels=True,
                node_color=node_colors, font_color="white", node_size=800, font_size=12)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title(f"Solution (cost={cost:.2f}) | Red: s=+1  Blue: s=-1")
        plt.tight_layout()
        plt.show()
