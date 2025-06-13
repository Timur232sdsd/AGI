from core.category import Object, Morphism, Category, NeuralMorphism  # Добавлен импорт NeuralMorphism
from core.functor import FuzzyFunctor
from core.topos import NeuralTopos
import jax.numpy as jnp
import torch
import torch.nn as nn

# 1. Create sensory processing category
sensory = Category("SensoryProcessing")
image = Object("Image")
feature_vector = Object("FeatureVector")
sensory.add_object(image)
sensory.add_object(feature_vector)

# Neural network for feature extraction
feature_extractor = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*14*14, 128)  # Исправлен расчет размера
)
feature_morph = NeuralMorphism("extract_features", image, feature_vector, feature_extractor)
sensory.add_morphism(feature_morph)

# 2. Create conceptual topos
conceptual = NeuralTopos("ConceptualSpace")  # Исправлено название
cat_concept = Object("CatConcept")
animal_concept = Object("AnimalConcept")
conceptual.add_object(cat_concept)
conceptual.add_object(animal_concept)

# Add symbolic rules
conceptual.add_rule("classify", "Implies(HasFur(x) & HasWhiskers(x), IsCat(x))")

# 3. Create functor between categories
F = FuzzyFunctor(
    source=sensory,
    target=conceptual,
    ob_map={
        image: cat_concept,
        feature_vector: animal_concept
    },
    hom_map={
        feature_morph: Morphism("classify", cat_concept, animal_concept)
    }
)

# 4. Define verification diagram
diagram = {
    "paths": [
        [feature_morph, F.hom_map[feature_morph]]
    ],
    "input": jnp.zeros((1, 3, 32, 32))  # Исправлены размеры
}

# 5. Train with cat/dog dataset
dataset = jnp.zeros((10, 3, 32, 32))  # Заглушка для данных
F.optimize(diagram, dataset)

# 6. Run inference on new image
new_image = jnp.zeros((1, 3, 32, 32))  # Заглушка для изображения
features = feature_morph(torch.from_numpy(new_image))  # Конвертация в тензор
concept = F.hom_map[feature_morph](features.detach().numpy())  # Конвертация обратно
classification = conceptual.neural_evaluate(
    conceptual.get_morphisms(cat_concept, animal_concept)[0], 
    concept
)

print(f"Classification result: {classification}")