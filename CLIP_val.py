import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Read labeled JSON file
with open('labeled_image_text_pairs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Set threshold
theta = 0.33
similarities = []
true_labels = []
predicted_labels = []
misclassified_samples = []

# Iterate over each image and text pair
for item in data:
    image_path = item['image']
    text = item['caption']
    true_label = item['label']  # Assuming the label is 1 for match and 0 for no match

    # Load the image using PIL
    image = Image.open(image_path)

    # Process image and text with truncation
    inputs = processor(
        text=[str(text)],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)

    # Get image and text embeddings
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # Calculate cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
    similarities.append(cosine_similarity.item())

    # Determine if they match
    is_match = cosine_similarity >= theta
    predicted_label = 1 if is_match else 0
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

    # Record misclassified samples
    if predicted_label != true_label:
        misclassified_samples.append({
            'image': image_path,
            'text': text,
            'true_label': true_label,
            'predicted_label': predicted_label
        })

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Visualize cosine similarity distribution
plt.hist(similarities, bins=50, alpha=0.75, color='blue')
plt.title('Cosine Similarity Distribution')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Visualize misclassified samples
misclassified_df = pd.DataFrame(misclassified_samples)
print(misclassified_df)
# Save misclassified samples to a CSV file
misclassified_df.to_csv('misclassified_samples1.csv', index=False, encoding='utf-8')
# Plot confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.matshow(conf_matrix, cmap='coolwarm')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()
