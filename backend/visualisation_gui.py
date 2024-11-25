import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import torch
import numpy as np

def tensor_to_pil(tensor, mean, std):
    # De-normalize the tensor
    tensor = tensor * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    tensor = tensor.clamp(0, 1)  # Ensure values are in [0, 1]
    tensor = tensor.cpu().squeeze()
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # Change to H, W, C
    return Image.fromarray((tensor.numpy() * 255).astype(np.uint8))


def create_images():
    # Simulate image creation (replace this with your actual tensor loading logic)
    return [tensor_to_pil(torch.rand(3, 200, 200)) for _ in range(4)]


def setup_gui(root, samples, predicted_prices, actual_prices):
    current_index = 0

    allways_show_predicted = False

    # Convert each sample (list of PIL images) into Tkinter-compatible images
    tk_samples = [[ImageTk.PhotoImage(image) for image in sample] for sample in samples]
    
    actual_label = Label(root, text=f"Real: ${actual_prices[current_index]:.2f}", font=("Helvetica", 25), bg='darkgray')
    actual_label.grid(row=0, column=0, columnspan=1, padx=10, pady=10)

    predicted_label = Label(root, text="Pred: ?", font=("Helvetica", 25), bg='darkgray')
    predicted_label.grid(row=0, column=1, columnspan=1)

    def show_prices():
        predicted_label.config(text=f"Pred: ${predicted_prices[current_index]:.2f}")

    show_button = Button(root, text="Show Prices", command=show_prices, padx=30, pady=15, font=("Helvetica", 13), bg='orange')
    show_button.grid(row=0, column=2, columnspan=1, padx=10, pady=10)

    def set_allways_show_predicted():
        nonlocal allways_show_predicted
        allways_show_predicted = not allways_show_predicted
        print(allways_show_predicted)

    show_all_checkbox = tk.Checkbutton(root, text="Allways show predictions", variable=allways_show_predicted, command=set_allways_show_predicted, padx=30, pady=15, font=("Helvetica", 13))
    show_all_checkbox.grid(row=0, column=3, columnspan=1, padx=10, pady=10)

    print(allways_show_predicted)


    image_labels = [Label(root) for _ in range(len(samples[0]))]
    for i, label in enumerate(image_labels):
        label.grid(row=1, column=i, padx=10, pady=10)


    def update_images():
        for i, label in enumerate(image_labels):
            label.config(image=tk_samples[current_index][i])


    def next_sample():
        nonlocal current_index
        current_index = (current_index + 1) % len(samples)
        update_images()
        if allways_show_predicted:
            predicted_label.config(text=f"Pred: ${predicted_prices[current_index]:.2f}")
        else:
            predicted_label.config(text="Pred: ?")
        actual_label.config(text=f"Real: ${actual_prices[current_index]:.2f}")

    def previous_sample():
        nonlocal current_index
        current_index = (current_index - 1) % len(samples)
        update_images()
        if allways_show_predicted:
            predicted_label.config(text=f"Pred: ${predicted_prices[current_index]:.2f}")
        else:
            predicted_label.config(text="Pred: ?")
        actual_label.config(text=f"Real: ${actual_prices[current_index]:.2f}")


    prev_button = Button(root, text="<<", command=previous_sample, padx=30, pady=20, font=("Helvetica", 18), fg='black')
    prev_button.grid(row=2, column=0, padx=10, pady=10, columnspan=3, sticky='w')

    next_button = Button(root, text=">>", command=next_sample, padx=30, pady=20, font=("Helvetica", 18), fg='black')
    next_button.grid(row=2, column=3, padx=10, pady=10, columnspan=3, sticky='e')

    # init
    update_images()


def create_samples():
    return [[tensor_to_pil(torch.rand(3, 200, 200)) for _ in range(4)] for _ in range(4)]


def visualise_results(samples, predicted_prices, actual_prices):
            # Main program
    root = tk.Tk()
    root.title("Tensor Image Gallery")

    # style
    root.configure(bg='darkgray'
                , padx=10, pady=10, relief=tk.RAISED, borderwidth=2)

    setup_gui(root, samples, predicted_prices, actual_prices)
    root.mainloop()

if __name__ == "__main__":
    predicted_prices = [150.00, 250.00, 350.00, 450.00]
    actual_prices = [140.00, 260.00, 330.00, 420.00]
    samples = create_samples()

    visualise_results(samples, predicted_prices, actual_prices)
    
