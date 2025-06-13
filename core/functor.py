import jax
import jax.numpy as jnp
import optax
from .category import Category, Object, Morphism
from typing import Dict, Callable, List
from itertools import combinations

class FuzzyFunctor:
    def __init__(self, source: Category, target: Category, 
                 ob_map: Dict[Object, Object], 
                 hom_map: Dict[Morphism, Morphism],
                 loss_fn: Callable = lambda diff: jnp.mean(jnp.square(diff))):  # Fixed: MSE loss implementation
        self.source = source
        self.target = target
        self.ob_map = ob_map
        self.hom_map = hom_map
        self.loss_fn = loss_fn
        self.params = {}
        
    def __call__(self, morph: Morphism, x: jnp.ndarray) -> jnp.ndarray:
        target_morph = self.hom_map[morph]
        return target_morph(x, self.params.get(morph.name, None))
    
    def commutativity_loss(self, diagram: Dict, data: jnp.ndarray) -> jnp.ndarray:
        losses = []
        paths = diagram["paths"]
        
        for path1, path2 in combinations(paths, 2):
            comp1 = self.compose_path(path1)
            comp2 = self.compose_path(path2)
            # Compute difference between path outputs
            diff = comp1(data) - comp2(data)
            # Apply loss to the difference
            losses.append(self.loss_fn(diff))
            
        return jnp.mean(jnp.array(losses))
    
    def compose_path(self, path: List[Morphism]) -> Callable:
        def composition(x):
            for morph in path:
                x = self.hom_map[morph](x)
            return x
        return composition
    
    def optimize(self, diagram: Dict, dataset: jnp.ndarray, 
                 steps: int = 100, lr: float = 0.01):
        opt = optax.adam(lr)
        opt_state = opt.init(self.params)
        
        def loss_fn(params):
            self.params = params
            return self.commutativity_loss(diagram, dataset)
        
        grad_fn = jax.value_and_grad(loss_fn)
        
        for _ in range(steps):
            loss, grads = grad_fn(self.params)
            updates, opt_state = opt.update(grads, opt_state)
            self.params = optax.apply_updates(self.params, updates)