import os
import torch
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import glob
import time
import csv
from GenreModel import load_model_and_tokenizer

#colors for UI
PRIMARY_COLOR = "#2E86C1"
SECONDARY_COLOR = "#AED6F1"
BUTTON_COLOR = "#2980B9"
TEXT_COLOR = "#FFFFFF"


class LandingPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        #prevent frame shrinking to content
        self.pack_propagate(False)

        #container frame for centering content
        centering_frame = tk.Frame(self)
        centering_frame.place(relx=0.5, rely=0.5, anchor='center')

        #title
        landing_title = tk.Label(centering_frame, text="Book Genre Classification Model", font=("Arial", 24, "bold"))
        landing_title.pack(pady=40)

        button_frame = tk.Frame(centering_frame)
        button_frame.pack(pady=20)

        summary_button = tk.Button(
            button_frame, text="Enter Summary & Predict Genre",
            font=("Segoe UI", 12, "bold"), bg=BUTTON_COLOR, fg="white", relief="raised", bd=2, width=30, command=lambda: controller.show_frame("GenreGenerator")
        )
        summary_button.grid(row=0, column=0, padx=20, pady=10)

        dashboard_button = tk.Button(
            button_frame, text="View Dashboard",
            font=("Segoe UI", 12, "bold"), bg=BUTTON_COLOR, fg="white", relief="raised", bd=2,
            width=30,
            command=lambda: controller.show_frame("DashboardPage")
        )
        dashboard_button.grid(row=1, column=0, padx=20, pady=10)

        #exit button
        exit_button = tk.Button(
            button_frame, text="Exit Application",
            font=("Segoe UI", 12, "bold"), bg=BUTTON_COLOR, fg="white", relief="raised", bd=2,
            width=30,
            command=self.controller.quit
        )
        exit_button.grid(row=2, column=0, padx=20, pady=10)

#chunks user input
def chunk_input(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    user_summary_chunking = []

    for i in range(0, len(tokens), stride):
        summary_chunk = tokens[i:i + max_length]
        if len(summary_chunk) < max_length:
            summary_chunk += [tokenizer.pad_token_id] * (max_length - len(summary_chunk))
        user_summary_chunking.append(summary_chunk)
        if i + max_length >= len(tokens):
            break

    user_input_ids = torch.tensor(user_summary_chunking).unsqueeze(0)  # shape: (1, num_chunks, max_length)
    attention_mask = (user_input_ids != tokenizer.pad_token_id).long()
    return user_input_ids, attention_mask


class GenreGenerator(tk.Frame):
    def __init__(self, parent, controller, model, tokenizer, genre_classes, device):
        super().__init__(parent)
        self.controller = controller
        self.model = model
        self.tokenizer = tokenizer
        self.genre_classes = genre_classes
        self.device = device

        #header for summary entry page
        label = tk.Label(self, text="Enter a book summary:", font=("Segoe UI", 12, "bold"), bg='white', bd=2)
        label.pack(pady=10)
        self.text_box = tk.Text(self, height=10, width=70)
        self.text_box.pack(pady=5)

        #button to get prediction
        predict_button = tk.Button(
            self, text="Predict Genre", command=self.predict_genre,
            font=("Segoe UI", 11, "bold"), bg='#2E86C1', fg='white', relief="raised", bd=2
        )
        predict_button.pack(pady=10)

        #back button to landing page
        back_button = tk.Button(
            self, text="Back to Home", command=lambda: controller.show_frame("LandingPage"),
            font=("Segoe UI", 11, "bold"), bg='#2E86C1', fg='white', relief="raised", bd=2
        )
        back_button.pack(pady=5)

        #Tkinter widget to display predictions
        self.result_label = tk.Label(self, text="", justify="left", font=("Segoe UI", 16, "bold"))
        self.result_label.pack(pady=10)

        #display word cloud image per top predicted genre
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)
        self.img_tk = None

    def predict_genre(self):
        #get user input with error if no summary is provided
        user_input = self.text_box.get("1.0", "end").strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a book summary.")
            return

        #tokenize the input
        user_input_ids, attention_mask = chunk_input(user_input, self.tokenizer)
        user_input_ids = user_input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        #run model in inference mode
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(input_ids=user_input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]  # Shape: (num_labels,)

        #gets top 3 highest probability genres
        top_indices = probs.argsort()[-3:][::-1]
        top_genres = [(self.genre_classes[i], probs[i] * 100) for i in top_indices]

        #display info in UI
        result_text = "Top 3 Predicted Genres:\n\n"
        for genre, confidence in top_genres:
            indicator = "✅" if confidence >= 75 else "⚠️"
            genre_title = genre.title()  # Capitalize each word
            result_text += f"{indicator} {genre_title}: {confidence:.1f}%\n"

        self.result_label.config(text=result_text)

        #display top probability genre's wordcloud image
        if top_genres:
            #building image path to search
            top_genre_name = top_genres[0][0]
            safe_genre_name = top_genre_name.replace(" ", "_").replace("/", "_")
            wordcloud_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordcloud_images")
            wordcloud_filename = f"wordcloud_{safe_genre_name}.png"
            image_path = os.path.join(wordcloud_directory, wordcloud_filename)

            #displays image if found
            if os.path.exists(image_path):
                try:
                    wordcloud_image = Image.open(image_path)
                    wordcloud_image = wordcloud_image.resize((900, 500), Image.LANCZOS)
                    self.img_tk = ImageTk.PhotoImage(wordcloud_image)
                    self.image_label.config(image=self.img_tk)
                except Exception as e:
                    messagebox.showerror("Image Error", f"Error loading image:\n{e}")
                    self.image_label.config(image="")
            else:
                self.image_label.config(image="")
                print("Image does not exist")
        else:
            self.image_label.config(image="")
            print("No genres confidently predicted.")

        #logging predictions to users, formatted as CSV
        try:
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_log.csv")
            file_exists = os.path.isfile(log_path)

            with open(log_path, mode='a', newline='', encoding='utf-8') as log_file:
                writer = csv.writer(log_file)

                #write header for new files
                if not file_exists:
                    writer.writerow([
                        "Timestamp", "Summary",
                        "Top Genre", "Top Confidence",
                        "Second Genre", "Second Confidence",
                        "Third Genre", "Third Confidence"
                    ])

                writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    user_input,
                    top_genres[0][0], f"{top_genres[0][1]:.1f}%",
                    top_genres[1][0], f"{top_genres[1][1]:.1f}%",
                    top_genres[2][0], f"{top_genres[2][1]:.1f}%"
                ])
        except Exception as e:
            print(f"Error writing to CSV log file: {e}")


class DashboardPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="white")

        #store directory of graph images
        graph_image_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_plot_images")
        self.image_paths = sorted(glob.glob(os.path.join(graph_image_directory, "*.png")) + glob.glob(os.path.join(graph_image_directory, "*.jpg")))

        #message if no images were found
        if not self.image_paths:
            label = tk.Label(self, text="No images found in 'graph_plot_images'", font=("Segoe UI", 12, "bold"), bg=BUTTON_COLOR, fg="white", relief="raised", bd=2)
            label.pack(pady=20)
            return

        self.image_names = [os.path.basename(p) for p in self.image_paths]
        self.image_dict = {os.path.basename(p): p for p in self.image_paths}

        self.current_index = tk.IntVar(value=0)
        self.zoom_level = tk.DoubleVar(value=1.0)
        self.original_img = None

        #dashboard functionality
        label = tk.Label(self, text="Select a Graph:", font=("Segoe UI", 12, "bold"), bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="raised", bd=2)
        label.pack(pady=(20, 5))
        self.selected_image = tk.StringVar()
        dropdown = ttk.Combobox(self, textvariable=self.selected_image, values=self.image_names, state="readonly", width=40)
        dropdown.pack(pady=5)
        dropdown.set("Choose an image")
        open_btn = tk.Button(self, text="Open Graph", command=self.open_selected_image, font=("Segoe UI", 11, "bold"), bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="raised", bd=2)
        open_btn.pack(pady=(0, 15))

        #canvas and scrollbars
        canvas_frame = tk.Frame(self, bg="white")
        canvas_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="white", width=700, height=450)
        self.canvas.pack(side="left", fill="both", expand=True)

        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.pack(side="right", fill="y")

        h_scroll = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        h_scroll.pack(fill="x", padx=10)

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.image_on_canvas = None

        #bottom control
        bottom_controls_frame = tk.Frame(self, bg="white")
        bottom_controls_frame.pack(pady=10)

        #navigation buttons
        nav_frame = tk.Frame(bottom_controls_frame, bg="white")
        nav_frame.grid(row=0, column=0, sticky="ew")

        prev_btn = tk.Button(nav_frame, text="← Previous", command=self.show_previous_image, font=("Segoe UI", 11, "bold"), bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="raised", bd=2)
        prev_btn.grid(row=0, column=0, padx=5, pady=5)

        next_btn = tk.Button(nav_frame, text="Next →", command=self.show_next_image, font=("Segoe UI", 11, "bold"), bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="raised", bd=2)
        next_btn.grid(row=0, column=1, padx=5, pady=5)

        #zoom buttons
        zoom_frame = tk.Frame(bottom_controls_frame, bg="white")
        zoom_frame.grid(row=1, column=0, sticky="ew")

        zoom_in_btn = tk.Button(zoom_frame, text="Zoom In +", command=self.zoom_in, font=("Segoe UI", 11, "bold"), bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="raised", bd=2)
        zoom_in_btn.grid(row=0, column=0, padx=5, pady=5)

        zoom_out_btn = tk.Button(zoom_frame, text="Zoom Out -", command=self.zoom_out, font=("Segoe UI", 11, "bold"), bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="raised", bd=2)
        zoom_out_btn.grid(row=0, column=1, padx=5, pady=5)

        #back to home button
        back_btn = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame("LandingPage"), font=("Segoe UI", 11, "bold"), bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="raised", bd=2)
        back_btn.pack(pady=10)

        #show first image
        self.update_image_display(self.current_index.get())

    def update_image_display(self, index):
        path = self.image_paths[index]
        self.original_img = Image.open(path)
        self.zoom_level.set(1.0)
        self.show_image_on_canvas()

    def show_image_on_canvas(self):
        if self.original_img is None:
            return
        zoom = self.zoom_level.get()
        width, height = self.original_img.size
        new_size = (int(width * zoom), int(height * zoom))
        image_resized = self.original_img.resize(new_size, Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(image_resized)

        #clear previous image
        self.canvas.delete("all")

        #add image to canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)

        #configure scrolling region
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
        index = self.current_index.get()
        if index < len(self.image_paths) - 1:
            self.current_index.set(index + 1)
            self.update_image_display(index + 1)

    def show_previous_image(self):
        index = self.current_index.get()
        if index > 0:
            self.current_index.set(index - 1)
            self.update_image_display(index - 1)

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

        #maximize window so it fills the screen but can still be moved/minimized
        self.state("zoomed")

        #lock window size
        self.resizable(False, False)

        #main container
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        #enable grid expansion
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        #load model and tokenizer
        model, tokenizer, genre_classes, device = load_model_and_tokenizer()

        #initialize all pages
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
