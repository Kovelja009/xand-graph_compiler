import json
from typing import Dict, Any, Union, List
import torch

from .graph import Graph, Node, Data, Operation, DataType
from .utils import load_config


class XandModule():
    def __init__(self, graph: Graph):
        self.graph = graph
        
    def forward(self):
        pass