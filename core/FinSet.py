from .category import Category, Object, Morphism, DataMorphism
import torch
import itertools

class FinSet(Category):
    def __init__(self, name: str = "FinSet"):
        super().__init__(name)
        self.cardinalities = {}
        
    def add_finite_object(self, name: str, size: int):
        obj = Object(name, properties={'size': size})
        self.add_object(obj)
        self.cardinalities[obj] = size
        return obj
    
    def add_function(self, name: str, src: Object, tgt: Object, mapping: list):
        if len(mapping) != self.cardinalities[src]:
            raise ValueError("Mapping size must match source cardinality")
        
        func = lambda x: mapping[x]
        return Morphism(name, src, tgt, func)
    
    def product(self, *objects: Object) -> Object:
        sizes = [self.cardinalities[obj] for obj in objects]
        total_size = 1
        for s in sizes:
            total_size *= s
            
        prod_obj = Object(
            "×".join(obj.name for obj in objects),
            properties={'size': total_size}
        )
        self.add_object(prod_obj)
        self.cardinalities[prod_obj] = total_size
        
        # Add projection morphisms
        for i, obj in enumerate(objects):
            proj_func = lambda x, i=i: x[i]
            proj = Morphism(
                f"π_{i}", 
                prod_obj, 
                obj,
                proj_func
            )
            self.add_morphism(proj)
        
        return prod_obj

    def entropy(self, morphism: Morphism, data: list) -> float:
        # Calculate entropy of a morphism given data
        pass

    def mutual_information(self, f: Morphism, g: Morphism, data: list) -> float:
        # Calculate mutual information between two morphisms
        pass