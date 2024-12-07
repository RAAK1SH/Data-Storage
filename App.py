import tkinter as tk
from customtkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import os
import pandas as pd
import webbrowser

# Knowing the data 
data_dir = 'datasets/test/'
classes = []
for folder in os.listdir(data_dir):
    classes.append(folder)
num_classes = len(classes)

df = pd.read_csv("./birds2.csv")

# loading the model architecture
model = torchvision.models.resnet50( weights="DEFAULT")

# freezing all the parameters from training
for param in model.parameters():
    param.require_grad = False

# adding a fc layer with relu activation and a dropout layer to prevent overfitting then output layer with num of classes
model.fc = nn.Sequential(nn.Linear(model.fc.in_features,1024),
                         nn.ReLU(),
                         nn.Dropout(0.3),
                         nn.Linear(1024,num_classes))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load your pre-trained model
model.load_state_dict(torch.load('./models/temp.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

# Define transformations to be applied to the input image
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Function to classify the image
def classify_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Apply transformations
    image = transform(image).unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        output = model(image)
    # Get the predicted class
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]  # Return the index of the predicted class

history = []

def display_infos(file_path,predicted_class):

    real_class = file_path.split('/', 20)[-1].split('.', 2)[0]

    # Update result label
    result_label.configure(text=f"This bird is a {predicted_class}.")

    # Display selected image
    display_image(file_path)

    # Display scientific name
    scientific_name = list(df.loc[(df["labels"]==predicted_class), ["scientific name"]]["scientific name"])[0]
    scientific_name_label.configure(text=f"The scientific name is {scientific_name}.")

    paths = list(df.loc[(df["labels"]==predicted_class), ["filepaths"]]["filepaths"])[0:5]

    link = list(df.loc[(df["labels"]==predicted_class), ["link"]]["link"])[0]
    if link != "No link" :
        link1.unbind("<Button 1>")
        link1.bind("<Button-1>", lambda e: callback(link))
        link1.configure(text="Visite Wikipedia")
    else :
        link1.unbind("<Button 1>")
        link1.configure(text="No link for now.")

    # Display other images related to the given image
    i = 1
    for path in paths :
        # Open the image
        image = Image.open("./datasets/"+ path)
        # Resize image to fit in the label
        image = image.resize((120, 120))
        # Convert image to PhotoImage
        img = CTkImage(dark_image=image,size=(120,120))
        # Update image label
        if i == 1:
            image_example1_label.configure(image=img)
        elif i == 2:
            image_example2_label.configure(image=img)
        elif i == 3:
            image_example3_label.configure(image=img)
        elif i == 4:
            image_example4_label.configure(image=img)

        i = i+1

    history.append({"file_path": file_path, "predicted_class": predicted_class, "real_class": real_class})


def show_history():
    history_window = CTkToplevel(root)
    history_window.title("Prediction History")
    history_window.geometry("500x300")
    test = CTkScrollableFrame(history_window)
    test.pack(fill="x")
    for entry in history:
        frame = CTkFrame(test)
        frame.pack(fill="x", padx=10, pady=5)
        
        img_label = CTkLabel(frame, text="")
        image = Image.open(entry["file_path"]).resize((50, 50))
        img = CTkImage(dark_image=image, size=(50, 50))
        img_label.configure(image=img)
        img_label.image = img 
        img_label.pack(side="left", padx=5, pady=5)
        
        info_label = CTkLabel(frame, text=f"Predicted: {entry['predicted_class']}, Real: {entry['real_class']}", font=("Arial", 12))
        info_label.pack(side="left", padx=5, pady=5)


def callback(url):
    webbrowser.open_new(url)

# Function to handle button click event for selecting image
def classify_button_click():
    # Open file dialog to select image
    file_path = filedialog.askopenfilename(initialdir=os.getcwd()+"imagesTest", title="Select JPEG File",
                                              filetypes=[("JPEG Files", "*.jpg")])
    if file_path:
        try:
            # Classify the image
            predicted_class = classify_image(file_path)
            display_infos(file_path,predicted_class)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

            result_label.configure(text="")
            scientific_name.configure(text="")
            display_image('./ressources/none.jpg')


# Function to display the selected image
def display_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Resize image to fit in the label
    image = image.resize((250, 250))
    # Convert image to PhotoImage
    img = CTkImage(
	dark_image=image,
	size=(250,250))
    # Update image label
    image_label.configure(image=img)

# Create main window
root = CTk()
root.title("Birds Classification App")

root.iconbitmap('ressources/logo.ico')

# Set window size
root.geometry("900x650")
root.resizable(False, False)

# Create a label to display the prediction result
result_label = CTkLabel(root, text="", font=("Arial ", 25, 'bold'))
result_label.pack(expand=True, pady=10)



# Create a label to display the selected image
image_label = CTkLabel(master=root, text="")
image_label.pack(pady=10)

# Create a button to select image
classify_button = CTkButton(root, text="Select Image", command=classify_button_click, corner_radius=32, 
                            fg_color="#4158D0", hover_color="#C850C0",
                            border_color="#FFCC70", border_width=2)
classify_button.pack()

frame = CTkFrame(root, fg_color="#cdb4db", border_color="#FFCC70", border_width=2)
frame.pack(expand=True,)

# Create a label to display the prediction result
scientific_name_label = CTkLabel(frame, text="", font=("Arial", 16, 'bold'), text_color="#000000")
scientific_name_label.pack(side="top", expand=True, padx=10, pady=10)

# Create a history button
history_button = CTkButton(root, text="Show History", command=show_history, corner_radius=32,
                            fg_color="#4158D0", hover_color="#C850C0",
                            border_color="#FFCC70", border_width=2)
history_button.pack(pady=10)

# Create labels to display more example of the prediction result
image_example1_label = CTkLabel(master=frame, text="")
image_example1_label.pack(side="left", expand=True, padx=10, pady=10)
image_example2_label = CTkLabel(master=frame, text="")
image_example2_label.pack(side="left", expand=True, padx=10, pady=10)
image_example3_label = CTkLabel(master=frame, text="")
image_example3_label.pack(side="right", expand=True, padx=10, pady=10)
image_example4_label = CTkLabel(master=frame, text="")
image_example4_label.pack(side="right", expand=True, padx=10, pady=10)


link1 = CTkLabel(frame, text="Visit Wikipedia", cursor="hand2", font=("Arial", 13, UNDERLINE), text_color="#4158D0")
link1.pack(side="bottom", expand=True, padx=10, pady=10)


#Initiate values as an example test
predicted_class = classify_image('imagesTest/avadavat.jpg')
display_infos('imagesTest/avadavat.jpg',predicted_class)


# Run the application
root.mainloop()