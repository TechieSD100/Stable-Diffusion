import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Text to Image Generator") 
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512, text="")
lmain.place(x=10, y=110)
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)


def generate():
    with autocast(device):
        result = pipe(prompt.get(), guidance_scale=8.5)  # Call the pipeline
        image = result.images[0]  # Access the first generated image

    image.save('generatedimage.png')  # Save the image locally
    img = ImageTk.PhotoImage(image)  # Convert to Tkinter-compatible image
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to avoid garbage collection



trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()