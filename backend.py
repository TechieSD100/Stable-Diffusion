import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
import os
import random

# Dataset Preparation
class TextToImageDataset(Dataset):
    """
    Custom dataset to simulate text-to-image training data.
    This class generates random image-text pairs for training.
    """
    def __init__(self, size=1000):
        self.size = size
        self.text_data = [f"Random caption {i}" for i in range(size)]
        self.image_data = torch.rand(size, 3, 64, 64)  # Dummy 64x64 RGB images
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        text = self.text_data[idx]
        image = self.image_data[idx]
        return {"text": text, "image": image}

# Model Architecture
class TextToImageModel(torch.nn.Module):
    """
    A dummy text-to-image model using a pretrained CLIP text encoder and a random image generator.
    """
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")  # Hugging Face CLIP model
        self.image_decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 3 * 64 * 64),  # 3 channels for 64x64 image
            torch.nn.Tanh()
        )
    
    def forward(self, text_inputs):
        # Text encoding
        text_features = self.text_encoder(**text_inputs).pooler_output
        # Generate image features
        image_features = self.image_decoder(text_features)
        # Reshape to image format
        images = image_features.view(-1, 3, 64, 64)
        return images

# Training Function
def train_model(model, dataloader, optimizer, epochs=10):
    """
    Function to simulate model training.
    """
    model.train()
    criterion = torch.nn.MSELoss()  # Using Mean Squared Error as dummy loss
    for epoch in range(epochs):
        for batch in dataloader:
            text_inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True).to(device)
            real_images = batch['image'].to(device)
            
            # Forward pass
            generated_images = model(text_inputs)
            
            # Compute loss
            loss = criterion(generated_images, real_images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Evaluation Function
def evaluate_model(model, text_prompts):
    """
    Generate images from text prompts.
    """
    model.eval()
    generated_images = []
    with torch.no_grad():
        for prompt in text_prompts:
            text_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            images = model(text_inputs).cpu()
            generated_images.append(images)
    return generated_images

# Integration with Hugging Face
def generate_using_huggingface(prompt, model_name="stabilityai/stable-diffusion-2"):
    """
    Use Hugging Face's pre-trained Stable Diffusion pipeline to generate images.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(device)
    image = pipeline(prompt).images[0]  # Generate image
    return image

# Main Workflow
if __name__ == "__main__":
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset and Dataloader
    dataset = TextToImageDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Tokenizer and Model
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = TextToImageModel().to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    print("Starting Training...")
    train_model(model, dataloader, optimizer, epochs=5)
    print("Training Complete!")
    
    # Evaluate the model
    prompts = ["A beautiful sunset over a mountain", "A futuristic city skyline"]
    print("Generating Images...")
    images = evaluate_model(model, prompts)
    
    # Integrate Hugging Face
    print("Generating with Hugging Face...")
    hf_image = generate_using_huggingface("A serene lake surrounded by trees")
    hf_image.show()
