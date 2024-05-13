GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like).

## Demo

https://huggingface.co/spaces/numind/NuZero

## Examples

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("numind/NuZero_token")
text = """
The Northern Lights, also known as the Aurora Borealis, is a natural light display in the sky, primarily seen in high-latitude regions like the Arctic. It occurs when charged particles from the sun interact with the Earth's magnetic field and collide with atoms in the upper atmosphere, causing them to emit light. The colors of the Northern Lights can vary from green to red, yellow, blue, and purple, depending on the type of gas particles colliding with the atmosphere.
"""
labels = ["color", "place", "physics"]
entities = model.predict_entities(text, labels)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

## Result


## Installation
To use this model, you must install the GLiNER Python library:
            ```
            !pip install gliner
            ```
         
## Usage
Once you've downloaded the GLiNER library, you can import the GLiNER class. You can then load this model using `GLiNER.from_pretrained` and predict entities with `predict_entities`.

