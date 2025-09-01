import os
import torch
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import glob

from GenreModel import load_model_and_tokenizer

'''class LandingPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Prevent frame shrinking to content
        self.pack_propagate(False)

        # Container frame to center content
        center_frame = tk.Frame(self)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        header = tk.Label(center_frame, text="Book Genre Classification Model", font=("Arial", 24, "bold"))
        header.pack(pady=40)

        btn_frame = tk.Frame(center_frame)
        btn_frame.pack(pady=20)

        summary_btn = tk.Button(
            btn_frame, text="Enter Summary & Predict Genre",
            font=("Arial", 14),
            width=30,
            command=lambda: controller.show_frame("GenreGenerator")
        )
        summary_btn.grid(row=0, column=0, padx=20, pady=10)

        dashboard_btn = tk.Button(
            btn_frame, text="View Dashboard",
            font=("Arial", 14),
            width=30,
            command=lambda: controller.show_frame("DashboardPage")
        )
        dashboard_btn.grid(row=1, column=0, padx=20, pady=10)'''

class LandingPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Prevent frame shrinking to content
        self.pack_propagate(False)

        # Container frame to center content
        center_frame = tk.Frame(self)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        header = tk.Label(center_frame, text="Book Genre Classification Model", font=("Arial", 24, "bold"))
        header.pack(pady=40)

        btn_frame = tk.Frame(center_frame)
        btn_frame.pack(pady=20)

        summary_btn = tk.Button(
            btn_frame, text="Enter Summary & Predict Genre",
            font=("Arial", 14),
            width=30,
            command=lambda: controller.show_frame("GenreGenerator")
        )
        summary_btn.grid(row=0, column=0, padx=20, pady=10)

        dashboard_btn = tk.Button(
            btn_frame, text="View Dashboard",
            font=("Arial", 14),
            width=30,
            command=lambda: controller.show_frame("DashboardPage")
        )
        dashboard_btn.grid(row=1, column=0, padx=20, pady=10)

        # 🔻 New Exit Button
        exit_btn = tk.Button(
            btn_frame, text="Exit Application",
            font=("Arial", 14),
            width=30,
            command=self.controller.quit  # This cleanly exits the Tkinter app
        )
        exit_btn.grid(row=2, column=0, padx=20, pady=10)


def chunk_input(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = []

    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) < max_length:
            chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))
        chunks.append(chunk)
        if i + max_length >= len(tokens):
            break

    input_ids = torch.tensor(chunks).unsqueeze(0)  # shape: (1, num_chunks, max_length)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask


class GenreGenerator(tk.Frame):
    def __init__(self, parent, controller, model, tokenizer, genre_classes, device):
        super().__init__(parent)
        self.controller = controller
        self.model = model
        self.tokenizer = tokenizer
        self.genre_classes = genre_classes
        self.device = device

        label = tk.Label(self, text="Enter a book summary:", font=("Arial", 14))
        label.pack(pady=10)

        self.text_box = tk.Text(self, height=10, width=70)
        self.text_box.pack(pady=5)

        predict_button = tk.Button(self, text="Predict Genre", command=self.predict_genre)
        predict_button.pack(pady=10)

        back_button = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame("LandingPage"))
        back_button.pack(pady=5)

        self.result_label = tk.Label(self, text="", font=("Arial", 12), justify="left")
        self.result_label.pack(pady=10)

        # Add image label (initially empty)
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        self.img_tk = None

    def predict_genre(self):
        user_input = self.text_box.get("1.0", "end").strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a book summary.")
            return

        input_ids, attention_mask = chunk_input(user_input, self.tokenizer)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).squeeze().tolist()

        if isinstance(predictions, bool):
            predictions = [predictions]

        predicted_genres = [genre for genre, pred in zip(self.genre_classes, predictions) if pred]

        result_text = "Predicted genre(s):\n\n"
        if predicted_genres:
            result_text += "\n".join(f"- {genre}" for genre in predicted_genres)
        else:
            result_text += "(No genres confidently predicted.)"

        self.result_label.config(text=result_text)

        # --- Load genre wordcloud image ---
        if predicted_genres:
            first_genre = predicted_genres[0]
            safe_genre_name = first_genre.replace(" ", "_").replace("/", "_")

            image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordcloud_images")
            image_filename = f"wordcloud_{safe_genre_name}.png"
            image_path = os.path.join(image_dir, image_filename)

            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    #set wordcloud image size
                    img = img.resize((1200, 600), Image.LANCZOS)
                    self.img_tk = ImageTk.PhotoImage(img)
                    self.image_label.config(image=self.img_tk)
                except Exception as e:
                    messagebox.showerror("Image Error", f"Error loading image:\n{e}")
                    self.image_label.config(image="")
            else:
                self.image_label.config(image="")  # Clear if image not found
                print("Image does not exist")
        else:
            self.image_label.config(image="")  # Clear image if no genres predicted
            print("No genres confidently predicted.")


class DashboardPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="white")

        img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_plot_images")
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))

        if not self.image_paths:
            label = tk.Label(self, text="No images found in 'graph_plot_images'", font=("Arial", 12), bg="white")
            label.pack(pady=20)
            return

        self.image_names = [os.path.basename(p) for p in self.image_paths]
        self.image_dict = {os.path.basename(p): p for p in self.image_paths}

        self.current_index = tk.IntVar(value=0)
        self.zoom_level = tk.DoubleVar(value=1.0)  # 1.0 = 100% zoom
        self.original_img = None  # To hold the original PIL image

        # UI Elements
        label = tk.Label(self, text="Select a Graph:", bg="white", font=("Arial", 12))
        label.pack(pady=(20, 5))

        self.selected_image = tk.StringVar()
        dropdown = ttk.Combobox(self, textvariable=self.selected_image, values=self.image_names, state="readonly", width=40)
        dropdown.pack(pady=5)
        dropdown.set("Choose an image")

        open_btn = tk.Button(self, text="Open Graph", command=self.open_selected_image)
        open_btn.pack(pady=(0, 15))

        # Create a fixed-size canvas with scrollbars to hold the image
        canvas_frame = tk.Frame(self, bg="white")
        canvas_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="white", width=700, height=450)  # Fixed size canvas
        self.canvas.pack(side="left", fill="both", expand=True)

        # Vertical scrollbar
        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.pack(side="right", fill="y")

        # Horizontal scrollbar
        h_scroll = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        h_scroll.pack(fill="x", padx=10)

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.image_on_canvas = None  # To hold the canvas image ID

        # Bottom controls frame for navigation and zoom buttons, centered and stacked
        bottom_controls_frame = tk.Frame(self, bg="white")
        bottom_controls_frame.pack(pady=10)

        # Navigation buttons row centered with smaller padding
        nav_frame = tk.Frame(bottom_controls_frame, bg="white")
        nav_frame.grid(row=0, column=0, sticky="ew")

        prev_btn = tk.Button(nav_frame, text="← Previous", command=self.show_previous_image)
        prev_btn.grid(row=0, column=0, padx=5, pady=5)

        next_btn = tk.Button(nav_frame, text="Next →", command=self.show_next_image)
        next_btn.grid(row=0, column=1, padx=5, pady=5)

        # Zoom buttons row centered below navigation buttons with smaller padding
        zoom_frame = tk.Frame(bottom_controls_frame, bg="white")
        zoom_frame.grid(row=1, column=0, sticky="ew")

        zoom_in_btn = tk.Button(zoom_frame, text="Zoom In +", command=self.zoom_in)
        zoom_in_btn.grid(row=0, column=0, padx=5, pady=5)

        zoom_out_btn = tk.Button(zoom_frame, text="Zoom Out -", command=self.zoom_out)
        zoom_out_btn.grid(row=0, column=1, padx=5, pady=5)

        # Back to home button below controls, centered
        back_btn = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame("LandingPage"))
        back_btn.pack(pady=10)

        self.update_image_display(self.current_index.get())

    def update_image_display(self, index):
        path = self.image_paths[index]
        self.original_img = Image.open(path)
        self.zoom_level.set(1.0)  # reset zoom on new image
        self.show_image_on_canvas()

    def show_image_on_canvas(self):
        if self.original_img is None:
            return
        zoom = self.zoom_level.get()
        width, height = self.original_img.size
        new_size = (int(width * zoom), int(height * zoom))
        img_resized = self.original_img.resize(new_size, Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(img_resized)  # Keep a reference!

        # Clear previous image
        self.canvas.delete("all")

        # Add image to canvas at top-left corner
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)

        # Configure scrolling region to image size
        self.canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))

    def open_selected_image(self):
        filename = self.selected_image.get()
        if filename in self.image_dict:
            index = self.image_names.index(filename)
            self.current_index.set(index)
            self.update_image_display(index)
        else:
            messagebox.showwarning("Warning", "Please select a valid image.")

    def show_next_image(self):
        idx = self.current_index.get()
        if idx < len(self.image_paths) - 1:
            self.current_index.set(idx + 1)
            self.update_image_display(idx + 1)

    def show_previous_image(self):
        idx = self.current_index.get()
        if idx > 0:
            self.current_index.set(idx - 1)
            self.update_image_display(idx - 1)

    def zoom_in(self):
        new_zoom = min(self.zoom_level.get() + 0.1, 3.0)
        self.zoom_level.set(new_zoom)
        self.show_image_on_canvas()

    def zoom_out(self):
        new_zoom = max(self.zoom_level.get() - 0.1, 0.1)
        self.zoom_level.set(new_zoom)
        self.show_image_on_canvas()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Book Genre Predictor")

        # Maximize window (not fullscreen) so it fills the screen but can still be moved/minimized
        self.state("zoomed")  # For Windows, on Linux may need: self.attributes('-zoomed', True)

        # Disable resizing (lock window size)
        self.resizable(False, False)

        # Main container frame
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        # Enable grid expansion inside container for proper layout
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        # Load model and tokenizer once
        model, tokenizer, genre_classes, device = load_model_and_tokenizer()

        # Initialize all application pages
        self.frames["LandingPage"] = LandingPage(container, self)
        self.frames["GenreGenerator"] = GenreGenerator(container, self, model, tokenizer, genre_classes, device)
        self.frames["DashboardPage"] = DashboardPage(container, self)

        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LandingPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()



if __name__ == "__main__":
    app = App()
    app.mainloop()
