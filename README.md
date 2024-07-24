# HelloBERT

HelloBERT is a fine-tuned BERT model for text classification, designed to identify greetings (`greeting`) and non-greetings (`not_greeting`) in both English and Spanish. The model is based on `bert-base-uncased`.

## Little Warning : )
> Nota: Es la primera vez que hice algo así, así que en el repo podés encontrar algunas notas sobre métricas y demás, bastante anotado porque es parte de mi aprendizaje sobre el tema : )
> Note: This is the first time I did something like this, so you will find some notes about metrics, part of my learning journey : )

## Intended Use

This model is intended for identifying whether a given text is a greeting or not. It can be used in various applications such as chatbots, automated customer service, and text analysis tools.

## Training Data

The model was fine-tuned on a custom dataset containing examples of greetings and non-greetings in English and Spanish. You can find the dataset [here](https://github.com/luzalbaposse/HelloBERT/tree/main). 

## Training Procedure

- **Base Model:** `bert-base-uncased`
- **Training Steps:** 3 epochs
- **Batch Size:** 16
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW
- **Evaluation Metric:** Accuracy

## How to Use

You can use this model directly with the Hugging Face `transformers` library as follows:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("luzalbaposse/HelloBERT")
tokenizer = AutoTokenizer.from_pretrained("luzalbaposse/HelloBERT")

# Create the pipeline
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Example usage
text = "Qué ondaaa??"
result = classifier(text)
print(f"Text: {text} => Label: {result[0]['label']}, Score: {result[0]['score']}")
```

## Limitations and Biases

While the model performs well on the training and validation sets, it may not generalize perfectly to all types of text inputs, especially those that are significantly different from the training data. Be cautious of potential biases in the training data that may affect model performance.

## Author

This model was fine-tuned by [Luz Alba Posse](https://huggingface.co/luzalbaposse).

## License

This model is licensed under the MIT License.
