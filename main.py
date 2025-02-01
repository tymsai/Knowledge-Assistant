# requirements.txt
"""
langchain==0.0.329
cohere==4.32
neo4j==5.14.1
torch==2.1.2
networkx==3.2.1
matplotlib==3.8.2
plotly==5.18.0
redis==5.0.1
rich==13.7.0
python-dotenv==1.0.0
"""

# knowledge_assistant.py
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import json
import redis
from functools import lru_cache
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from neo4j import GraphDatabase
from langchain.llms import Cohere
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import CohereEmbeddings

# Load environment variables
load_dotenv()

class Config:
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

class NeuralModels:
    class TransformerEncoder(nn.Module):
        def __init__(self, input_dim=768, nhead=8, num_layers=3):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=2048,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        def forward(self, x):
            return self.transformer(x)

    class GraphNetwork(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=384):
            super().__init__()
            self.conv1 = nn.Linear(input_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, hidden_dim)
            
        def forward(self, x, adj_matrix):
            x = torch.relu(self.conv1(torch.matmul(adj_matrix, x)))
            return torch.relu(self.conv2(torch.matmul(adj_matrix, x)))

class Cache:
    def __init__(self, host: str = Config.REDIS_HOST, port: int = Config.REDIS_PORT):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = 3600  # 1 hour cache lifetime

    def get(self, key: str) -> Optional[dict]:
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def set(self, key: str, value: dict):
        self.redis.setex(key, self.ttl, json.dumps(value))

class KnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_indexes(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.content)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Query) ON (n.content)")

    def add_relationship(self, source: str, target: str, relationship: str):
        with self.driver.session() as session:
            session.run("""
                MERGE (s:Concept {content: $source})
                MERGE (t:Concept {content: $target})
                MERGE (s)-[:RELATES {type: $relationship}]->(t)
            """, source=source, target=target, relationship=relationship)

class Visualizer:
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def create_network(self) -> nx.Graph:
        G = nx.Graph()
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]-(m)
                RETURN n.content, type(r), m.content
            """)
            for record in result:
                G.add_edge(record['n.content'], record['m.content'], 
                          type=record['type(r)'])
        return G

    def plot_static(self, filename: str = 'knowledge_graph.png'):
        G = self.create_network()
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
               node_size=1500, font_size=8)
        plt.savefig(filename)
        plt.close()

    def plot_interactive(self) -> go.Figure:
        G = self.create_network()
        pos = nx.spring_layout(G, dim=3)
        
        edge_traces = []
        node_trace = go.Scatter3d(
            x=[], y=[], z=[], mode='markers+text',
            hoverinfo='text', marker=dict(size=10, color='lightblue')
        )

        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_traces.append(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode='lines', line=dict(width=1, color='#888')
            ))

        for node in G.nodes():
            x, y, z = pos[node]
            node_trace.x = node_trace.x + (x,)
            node_trace.y = node_trace.y + (y,)
            node_trace.z = node_trace.z + (z,)

        return go.Figure(data=edge_traces + [node_trace])

class Assistant:
    def __init__(self):
        self.console = Console()
        self.cache = Cache()
        self.graph = KnowledgeGraph(
            Config.NEO4J_URI,
            Config.NEO4J_USER,
            Config.NEO4J_PASSWORD
        )
        self.visualizer = Visualizer(self.graph)
        self.cohere = Cohere()
        self.embeddings = CohereEmbeddings(cohere_api_key=Config.COHERE_API_KEY)
        self.history = []
        
        # Initialize neural models
        self.transformer = NeuralModels.TransformerEncoder()
        self.graph_network = NeuralModels.GraphNetwork()
        
        # Create database indexes
        self.graph.create_indexes()

    def start(self):
        self.console.print(Panel.fit(
            "🤖 Welcome to the Knowledge Graph AI Assistant!\n"
            "Type 'help' for available commands or 'exit' to quit.",
            title="AI Assistant",
            border_style="blue"
        ))

        while True:
            try:
                query = self.console.input("\n[bold blue]Query:[/] ").strip()
                
                if not query:
                    continue
                    
                if query.lower() == 'exit':
                    self.cleanup()
                    break
                    
                self.process_command(query)
                
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/]")

    def process_command(self, query: str):
        commands = {
            'help': self.show_help,
            'viz': self.show_visualization,
            'stats': self.show_stats,
            'history': self.show_history,
            'clear': self.clear_history
        }
        
        cmd = query.lower()
        if cmd in commands:
            commands[cmd]()
        else:
            self.process_query(query)

    def process_query(self, query: str):
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=100)
            
            # Check cache
            cache_key = f"query:{hash(query)}"
            cached_response = self.cache.get(cache_key)
            
            if cached_response:
                progress.update(task, completed=100)
                self.display_response(cached_response)
                return

            # Get response from Cohere
            progress.update(task, advance=30)
            response = self.get_ai_response(query)
            
            # Update knowledge graph
            progress.update(task, advance=30)
            self.update_graph(query, response)
            
            # Cache response
            progress.update(task, advance=30)
            self.cache.set(cache_key, response)
            
            # Store in history
            self.history.append((query, response))
            
            progress.update(task, completed=100)
            self.display_response(response)

    def get_ai_response(self, query: str) -> dict:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Please provide a detailed response to: {query}"
        )
        chain = LLMChain(llm=self.cohere, prompt=prompt)
        response = chain.run(query=query)
        
        return {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

    def update_graph(self, query: str, response: dict):
        # Add query-response relationship
        self.graph.add_relationship(
            query,
            response['response'][:100],  # Use first 100 chars as identifier
            "HAS_RESPONSE"
        )

    def display_response(self, response: dict):
        self.console.print("\n[bold green]Response:[/]")
        self.console.print(Panel(response['response'], border_style="green"))

    def show_help(self):
        help_text = """
        Available commands:
        - help: Show this help message
        - viz: Visualize the knowledge graph
        - stats: Show system statistics
        - history: Show query history
        - clear: Clear query history
        - exit: Exit the assistant
        
        Or simply type your question for an AI response!
        """
        self.console.print(Panel(help_text, title="Help", border_style="blue"))

    def show_visualization(self):
        with Progress() as progress:
            task = progress.add_task("[cyan]Generating visualizations...", total=100)
            
            # Generate static visualization
            progress.update(task, advance=50)
            self.visualizer.plot_static()
            
            # Generate interactive visualization
            progress.update(task, advance=40)
            fig = self.visualizer.plot_interactive()
            fig.show()
            
            progress.update(task, completed=100)

    def show_stats(self):
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (n) 
                RETURN count(n) as nodes,
                       size([(n)-[r]->(m) | r]) as relationships
            """)
            stats = result.single()
        
        stats_text = f"""
        System Statistics:
        • Nodes in Knowledge Graph: {stats['nodes']}
        • Relationships: {stats['relationships']}
        • Queries in History: {len(self.history)}
        • Cache Entries: {self.cache.redis.dbsize()}
        """
        self.console.print(Panel(stats_text, title="Statistics", border_style="cyan"))

    def show_history(self):
        if not self.history:
            self.console.print("[yellow]No history available[/]")
            return
            
        for i, (query, response) in enumerate(self.history, 1):
            self.console.print(f"\n[cyan]Query {i}:[/] {query}")
            self.console.print(f"[green]Response:[/] {response['response'][:100]}...")

    def clear_history(self):
        self.history.clear()
        self.console.print("[green]History cleared[/]")

    def cleanup(self):
        self.graph.close()
        self.console.print("[yellow]Goodbye! 👋[/]")

def main():
    if not Config.COHERE_API_KEY:
        print("Error: COHERE_API_KEY not found in environment variables")
        print("Please create a .env file with your credentials")
        return
        
    assistant = Assistant()
    assistant.start()

if __name__ == "__main__":
    main()
