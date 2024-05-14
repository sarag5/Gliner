from gliner import GLiNER

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Sample text for entity prediction
text = """
The Northern Lights, also known as the Aurora Borealis, is a natural light display in the sky, primarily seen in high-latitude regions like the Arctic. It occurs when charged particles from the sun interact with the Earth's magnetic field and collide with atoms in the upper atmosphere, causing them to emit light. The colors of the Northern Lights can vary from green to red, yellow, blue, and purple, depending on the type of gas particles colliding with the atmosphere.
"""

# Labels for entity prediction
labels = ["Person", "Award", "Date", "Competitions", "Teams"] # for v2.1 use capital case for better performance

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
