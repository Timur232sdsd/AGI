from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Dict, Set, Callable, Any, Optional, Union
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Экспортируемые классы
__all__ = [
    'Object', 
    'Morphism', 
    'DataMorphism', 
    'KernelMorphism',
    'NeuralMorphism',
    'SymbolicMorphism',  # Добавлен новый класс
    'Category'
]
class Object:
    def __init__(self, name: str, properties: Optional[dict] = None, 
                 metric: Callable = lambda x, y: x == y):
        self.name = name
        self.properties = properties or {}
        self.metric = metric
        self.embedding = None  # For neural-symbolic integration
        
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Obj({self.name})"
    
    def __eq__(self, other) -> bool:
        return self.name == other.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __call__(self, x, y) -> bool:
        if isinstance(x, list):
            return [self.metric(a, b) for a, b in zip(x, y)]
        return self.metric(x, y)

class Morphism(ABC):
    def __init__(self, name: str, src: Object, tgt: Object, 
                 func: Optional[Callable] = None):
        self.name = name
        self.func = func
        self.src = src
        self.tgt = tgt
        self.special: Dict[str, 'Morphism'] = {}
        self.params = {}  # For optimizable morphisms
        
    def __call__(self, x=None):
        if self.func is None:
            return x
        return self.func(x)
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Mor({self.src}→{self.tgt})"
    
    def compose(self, *morphisms: 'Morphism') -> 'Morphism':
        funcs = [self] + list(morphisms)
        name = "∘".join(m.name for m in funcs)
        return Morphism(
            name, 
            funcs[0].src, 
            funcs[-1].tgt,
            lambda x: reduce(lambda a, f: f(a), funcs, x)
        )
        
    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass

class DataMorphism(Morphism):
    def __init__(self, name: str, data: Any, tgt: Object):
        super().__init__(name, None, tgt, lambda: data)
        self.data = data
        
    def __call__(self, index=None):
        return self.data if index is None else self.data[index]
    
    def forward(self, x: Any) -> Any:
        return self.data[x] if hasattr(self.data, '__getitem__') else self.data

class KernelMorphism(Morphism):
    def __init__(self, name: str, src_type: type, tgt_type: type, 
                 func: Optional[Callable] = None):
        super().__init__(name, Object(src_type.__name__), Object(tgt_type.__name__), func)
        self.src_type = src_type
        self.tgt_type = tgt_type
        
    def __call__(self, x):
        if not isinstance(x, self.src_type):
            raise TypeError(f"Expected {self.src_type}, got {type(x)}")
        result = super().__call__(x)
        if not isinstance(result, self.tgt_type):
            raise TypeError(f"Expected {self.tgt_type}, got {type(result)}")
        return result
    
    def forward(self, x: Any) -> Any:
        return self.func(x)

# Добавлен недостающий класс NeuralMorphism
class NeuralMorphism(Morphism):
    def __init__(self, name: str, src: Object, tgt: Object, 
                 model: torch.nn.Module):
        super().__init__(name, src, tgt)
        self.model = model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Category:
    def __init__(self, name: str, 
                 objects: Optional[List[Object]] = None, 
                 morphisms: Optional[List[Morphism]] = None):
        self.name = name
        self.os: Set[Object] = set()
        self.ms: Dict[Object, Dict[Object, List[Morphism]]] = {}
        self.inputs: List[DataMorphism] = []
        self.kernel_morphisms: List[KernelMorphism] = []
        self.neural_morphisms: List[NeuralMorphism] = []  # Добавлено для NeuralMorphism
        
        for obj in (objects or []):
            self.add_object(obj)
        for mor in (morphisms or []):
            self.add_morphism(mor)

    def add_object(self, obj: Object):
        self.os.add(obj)

    def add_morphism(self, morph: Morphism):
        if morph.src not in self.os:
            self.add_object(morph.src)
        if morph.tgt not in self.os:
            self.add_object(morph.tgt)
            
        if morph.src not in self.ms:
            self.ms[morph.src] = {}
        if morph.tgt not in self.ms[morph.src]:
            self.ms[morph.src][morph.tgt] = []
            
        self.ms[morph.src][morph.tgt].append(morph)
        
        if isinstance(morph, DataMorphism):
            self.inputs.append(morph)
        elif isinstance(morph, KernelMorphism):
            self.kernel_morphisms.append(morph)
        elif isinstance(morph, NeuralMorphism):  # Добавлено для NeuralMorphism
            self.neural_morphisms.append(morph)
    
    def get_morphisms(self, src: Object, tgt: Object) -> List[Morphism]:
        return self.ms.get(src, {}).get(tgt, [])
    
    def get_paths(self, src: Object, tgt: Object) -> List[List[Morphism]]:
        # Path finding using BFS
        queue = [(src, [])]
        paths = []
        
        while queue:
            current, path = queue.pop(0)
            
            if current == tgt:
                paths.append(path)
                continue
                
            if current in self.ms:
                for neighbor, morphisms in self.ms[current].items():
                    for morph in morphisms:
                        new_path = path + [morph]
                        queue.append((neighbor, new_path))
        
        return paths
    
    def diagram_commutativity(self, diagram: Dict) -> float:
        paths = diagram["paths"]
        results = []
        
        for path in paths:
            composition = self.compose_path(path)
            results.append(composition(diagram["input"]))
        
        # Calculate variance between results
        return torch.var(torch.stack(results)).item()
    
    def compose_path(self, path: List[Morphism]) -> Callable:
        def composed(x):
            for morph in path:
                x = morph(x)
            return x
        return composed
    
    def draw(self):
        G = nx.DiGraph()
        color_map = {
            Morphism: 'black',
            KernelMorphism: 'blue',
            DataMorphism: 'red',
            NeuralMorphism: 'green'  # Добавлено для NeuralMorphism
        }
        
        for obj in self.os:
            G.add_node(obj.name)
        
        for src, targets in self.ms.items():
            for tgt, morphisms in targets.items():
                for mor in morphisms:
                    G.add_edge(
                        src.name, 
                        tgt.name, 
                        label=mor.name,
                        color=color_map.get(type(mor), 'gray')
                    )
        
        pos = nx.spring_layout(G)
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
        
        nx.draw(G, pos, with_labels=True, edge_color=edge_colors, 
                font_weight='bold', node_size=700)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title(f"Category: {self.name}")
        plt.show()
class SymbolicMorphism(Morphism):
    def __init__(self, name: str, src: Object, tgt: Object, 
                 rule: Optional[Any] = None):
        super().__init__(name, src, tgt)
        self.rule = rule
        self.predicates = self.extract_predicates() if rule else {}
        
    def extract_predicates(self) -> Dict[str, Callable]:
        """Извлекает предикаты из символьного правила (заглушка)"""
        # В реальной реализации здесь должен быть парсинг rule
        return {
            'HasFur': lambda x: x[0] > 0.5,
            'HasWhiskers': lambda x: x[1] > 0.5,
            'IsCat': lambda x: x.mean() > 0.7
        }
        
    def forward(self, x: Any) -> Any:
        """Применяет символьное правило к входным данным"""
        if self.rule is None:
            return 0
        
        # Вычисляем значения предикатов
        values = {}
        for name, pred in self.predicates.items():
            try:
                values[name] = pred(x)
            except Exception as e:
                print(f"Error evaluating predicate {name}: {e}")
                values[name] = False
        
        # Упрощенная логика применения правила
        if values.get('HasFur', False) and values.get('HasWhiskers', False):
            return values.get('IsCat', 0)
        return 0
    
    def __call__(self, x=None):
        return self.forward(x)