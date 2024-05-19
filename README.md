# Production_Lines
Reference code for feature request for Minecraft mod EMI.

Some python code that through given list of recipes (with types of resources: item, fluid, etc.), restrictions (which processing methods are allowed, which resources are allowed, max speed of output for resoucre type (restriction for pipes)), and requirements (resource, amount, needed processing time) will generate Obsidian canvas with stages of crafting scheme with best amount of instances of method (machine, etc.) for given needed time.

The code is NOT PERFECT, but it works, and give the idea.

The format of recipe, which i used here is:

```python
"name of recipe": {
      "foto": "name of recipe.png", #  Maybe unnecessary thing, just to add for each recipe a image, i think can be implemented in better way.
      "input_combinations": [
        [{ "amount": amount, "name": "name of resource" }, ...]
        ... # Recipe can have multiple input combinations and only one output. Was needed for situations where through the same machine, at the same time, you can craft some resources from different input combiantions.
      ],
      "method": "method through which recipe is going (Machine name, etc.)",
      "outputs": [
        {
          "amount": amount,
          "name": "name of resource",
          "probability": Probability to craft, #  Hardcoded that if probability < 1 then for each working of recipe, output is multiplied by probability * 0.8, just to make sure that it will craft needed amount of it. 
          "type": ["type"] #  List of types (for pipes speed restriction)
        }
      ],
      "time": time #  time to process recipe (i used ticks)
    }
```
