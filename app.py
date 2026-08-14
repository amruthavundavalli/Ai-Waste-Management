import streamlit as st
from ultralytics import YOLO
import cv2
model = YOLO("yolo11n.pt")

def select_image():
    file = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if file:
        results = model.predict(source=file, device="cpu")
        result = results[0]

        if result.boxes:
            names = result.names
            classes = result.boxes.cls.tolist()
            detected = [names[int(c)] for c in classes]

            messagebox.showinfo(
                "AI Waste Detection",
                "Detected Waste:\n\n" + "\n".join(detected)
            )
        else:
            messagebox.showinfo(
                "AI Waste Detection",
                "No waste detected"
            )

root = tk.Tk()
root.title("AI Waste Management")
root.geometry("500x300")

title = tk.Label(
    root,
    text="AI WASTE MANAGEMENT",
    font=("Arial", 22, "bold")
)
title.pack(pady=40)

button = tk.Button(
    root,
    text="SELECT WASTE IMAGE",
    font=("Arial", 16),
    command=select_image
)
button.pack(pady=30)

root.mainloop()
