import numpy as np
from core.category import Object, Morphism, Category, NeuralMorphism, SymbolicMorphism
from core.functor import FuzzyFunctor
from core.topos import NeuralTopos
import jax.numpy as jnp
import torch
import torch.nn as nn
import sympy as sp  # Добавлен импорт sympy

# 1. Create sensory processing category
sensory = Category("SensoryProcessing")
image = Object("Image")
feature_vector = Object("FeatureVector")
sensory.add_object(image)
sensory.add_object(feature_vector)

# Исправленная нейросетевая архитектура
feature_extractor = nn.Sequential(
    nn.Conv2d(3, 16, 3),      # (32-3+1)=30 -> [16, 30, 30]
    nn.ReLU(),
    nn.MaxPool2d(2),           # 30//2=15 -> [16, 15, 15]
    nn.Flatten(),              # 16*15*15=3600
    nn.Linear(3600, 128)       # Правильный размер входа
)
feature_morph = NeuralMorphism("extract_features", image, feature_vector, feature_extractor)
sensory.add_morphism(feature_morph)

# 2. Create conceptual topos
conceptual = NeuralTopos("ConceptualSpace")
cat_concept = Object("CatConcept")
animal_concept = Object("AnimalConcept")
conceptual.add_object(cat_concept)
conceptual.add_object(animal_concept)

# Добавление правила с исправленным синтаксисом и словарем символов
conceptual.add_rule(
    "classify", 
    "Implies(And(HasFur(x), HasWhiskers(x)), IsCat(x))",  # Исправлены скобки и опечатка
    local_dict={
        'HasFur': sp.Function('HasFur'),
        'HasWhiskers': sp.Function('HasWhiskers'),
        'IsCat': sp.Function('IsCat')
    }
)

# Создаем символический морфизм для классификации
classify_morphism = SymbolicMorphism(
    "classify", 
    cat_concept, 
    animal_concept,
    rule=conceptual.rules["classify"]
)
conceptual.add_morphism(classify_morphism)

# 3. Create functor between categories
F = FuzzyFunctor(
    source=sensory,
    target=conceptual,
    ob_map={
        image: cat_concept,
        feature_vector: animal_concept
    },
    hom_map={
        feature_morph: classify_morphism
    }
)

# 4. Обработка тестового изображения
new_image = jnp.zeros((1, 3, 32, 32))  # Тестовый образ

# Преобразование JAX-массива -> NumPy -> Torch Tensor
new_image_np = np.array(new_image)
features = feature_morph.forward(torch.from_numpy(new_image_np))

# Преобразование Torch Tensor -> NumPy для SymbolicMorphism
features_np = features.detach().numpy()
concept = F.hom_map[feature_morph].forward(features_np)

print(f"Classification result: {concept}")