import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.quantization import quantize_dynamic

class SentimentLLM:
    def __init__(self, device='cpu'):
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self.model.to(self.device)
        # Quantize model for faster inference and reduced memory
        self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        print("Model quantized and ready.")

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return ["negative", "positive"][pred]

if __name__ == "__main__":
    llm = SentimentLLM()
    print(llm.classify("I love open source AI!"))
    print(llm.classify("This code is buggy and slow."))