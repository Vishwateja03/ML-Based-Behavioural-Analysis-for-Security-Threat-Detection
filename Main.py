import customtkinter as ctk
from customtkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import io
import webbrowser
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import pickle

# Set appearance
ctk.set_appearance_mode("Dark") 
ctk.set_default_color_theme("blue")

class IDS(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Main window
        self.title("ML-Based Threat Detection")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        # Add this to your __init__ method:
        self.best_accuracy = 0.0
        
        # Load and set icon
        try:
            self.iconbitmap("shield_icon.ico")  # Provide your icon file
        except:
            pass
        
        # Create main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.create_main_content()
        
        # Initialize variables
        self.dataset = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.spc_cnn = None
        self.labels = None
        self.le1 = LabelEncoder()
        self.le2 = LabelEncoder()
        self.le3 = LabelEncoder()
        self.le4 = LabelEncoder()
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.fscore = []
        self.current_tab = "data"
        
        # Load tutorial images
        self.load_tutorial_images()
        
    def load_tutorial_images(self):
        """Load tutorial images (placeholder - replace with actual images)"""
        try:
            # These would be your actual image paths
            self.tutorial_images = {
                "upload": ImageTk.PhotoImage(Image.open("upload_tutorial.png").resize((400, 250))),
                "preprocess": ImageTk.PhotoImage(Image.open("preprocess_tutorial.png").resize((400, 250))),
                "model": ImageTk.PhotoImage(Image.open("model_tutorial.png").resize((400, 250)))
            }
        except:
            # Create placeholder images if real ones aren't found
            self.tutorial_images = {
                "upload": self.create_placeholder_image("Upload Dataset Tutorial"),
                "preprocess": self.create_placeholder_image("Preprocessing Tutorial"),
                "model": self.create_placeholder_image("Model Training Tutorial")
            }
    
    def create_placeholder_image(self, text):
        """Create a placeholder image with text"""
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 250), color=(50, 50, 50))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        d.text((50, 100), text, fill=(255, 255, 255), font=font)
        return ImageTk.PhotoImage(img)
    
    def create_sidebar(self):
        """Create the sidebar with navigation and info"""
        sidebar = ctk.CTkFrame(self.main_container, width=200, corner_radius=0)
        sidebar.pack(side="left", fill="y", padx=(0, 10), pady=0)
        
        # Logo and title
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(pady=(10, 20), padx=10, fill="x")
        
        # Logo placeholder - replace with actual logo image
        try:
            logo_img = ImageTk.PhotoImage(Image.open("logo.png").resize((180, 60)))
            logo_label = ctk.CTkLabel(logo_frame, image=logo_img, text="")
            logo_label.image = logo_img
            logo_label.pack()
        except:
            ctk.CTkLabel(logo_frame, text="Dashboard", 
                        font=("Arial", 18, "bold")).pack()
        
        # Navigation buttons
        nav_options = [
            ("Data Processing", "data"),
            ("Model Training", "model"),
            ("Results & Testing", "results"),
            ("System Dashboard", "dashboard")
        ]
        
        for text, tab in nav_options:
            btn = ctk.CTkButton(sidebar, text=text, command=lambda t=tab: self.show_tab(t),
                               fg_color="transparent", hover_color=("#3A7EBF", "#1F538D"),
                               anchor="w", height=40)
            btn.pack(fill="x", pady=2)
        
        # Add some separation
        ctk.CTkLabel(sidebar, text="", height=20).pack()
        
        # System info
        info_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        info_frame.pack(fill="x", padx=5, pady=10)
        ctk.CTkLabel(info_frame, text="System Status:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        self.status_labels = {
            "data": ctk.CTkLabel(info_frame, text="Data: Not Loaded", text_color="red", anchor="w"),
            "model": ctk.CTkLabel(info_frame, text="Model: Not Trained", text_color="red", anchor="w"),
            "memory": ctk.CTkLabel(info_frame, text="Memory: Stable", text_color="green", anchor="w")
        }
        
        for label in self.status_labels.values():
            label.pack(fill="x", pady=2)
        
        # Tutorial button
        ctk.CTkButton(sidebar, text="Quick Tutorial", command=self.show_tutorial,
                     fg_color="#2D8BBA", hover_color="#266A8E").pack(side="bottom", pady=10)
    
    def create_main_content(self):
        """Create the main content area with tabs"""
        self.main_content = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.main_content.pack(side="right", fill="both", expand=True)
        
        # Create tab containers (initially hidden)
        self.tabs = {
            "data": self.create_data_tab(),
            "model": self.create_model_tab(),
            "results": self.create_results_tab(),
            "dashboard": self.create_dashboard_tab()
        }
        
        # Show default tab
        self.show_tab("data")
    
    def create_data_tab(self):
        """Create the data processing tab"""
        tab = ctk.CTkFrame(self.main_content, fg_color="transparent")
        
        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header, text="Data Processing", font=("Arial", 20, "bold")).pack(side="left")
        
        # Main content area
        content = ctk.CTkFrame(tab)
        content.pack(fill="both", expand=True)
        
        # Left panel - buttons and info
        left_panel = ctk.CTkFrame(content, width=250)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Action buttons
        actions = [
            ("Upload Dataset", self.uploadDataset, "#5D9B9A"),
            ("Preprocess Data", self.preprocess, "#8A6EB8"),
            ("ADASYN Augmentation", self.augmentation, "#D35B58"),
            ("Train/Test Split", self.trainTestSplit, "#E1A730")
        ]
        
        for text, command, color in actions:
            btn = ctk.CTkButton(left_panel, text=text, command=command,
                               fg_color=color, hover_color=self.darken_color(color),
                               height=40, corner_radius=6)
            btn.pack(fill="x", pady=5)
        
        # Data info panel
        info_frame = ctk.CTkFrame(left_panel)
        info_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(info_frame, text="Dataset Info", font=("Arial", 14, "bold")).pack()
        
        self.data_info = {
            "samples": ctk.CTkLabel(info_frame, text="Samples: 0", anchor="w"),
            "features": ctk.CTkLabel(info_frame, text="Features: 0", anchor="w"),
            "classes": ctk.CTkLabel(info_frame, text="Classes: 0", anchor="w"),
            "status": ctk.CTkLabel(info_frame, text="Status: Not processed", anchor="w")
        }
        
        for label in self.data_info.values():
            label.pack(fill="x", pady=2)
        
        # Right panel - output and visualizations
        right_panel = ctk.CTkFrame(content)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Notebook for output and plots
        self.data_notebook = ctk.CTkTabview(right_panel)
        self.data_notebook.pack(fill="both", expand=True)
        
        # Console output tab
        self.data_console = self.data_notebook.add("Console")
        self.data_text = ctk.CTkTextbox(self.data_console, wrap="word", font=("Consolas", 12))
        self.data_text.pack(fill="both", expand=True)
        
        # Visualization tab
        self.data_viz = self.data_notebook.add("Visualizations")
        self.data_viz_frame = ctk.CTkFrame(self.data_viz, fg_color="transparent")
        self.data_viz_frame.pack(fill="both", expand=True)
        
        # Initially empty visualization area
        self.data_viz_label = ctk.CTkLabel(self.data_viz_frame, text="Visualizations will appear here",
                                          font=("Arial", 14))
        self.data_viz_label.pack(pady=50)
        
        return tab
    
    def create_model_tab(self):
        """Create the model training tab"""
        tab = ctk.CTkFrame(self.main_content, fg_color="transparent")
        
        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header, text="Model Training", font=("Arial", 20, "bold")).pack(side="left")
        
        # Main content area
        content = ctk.CTkFrame(tab)
        content.pack(fill="both", expand=True)
        
        # Left panel - buttons and info
        left_panel = ctk.CTkFrame(content, width=250)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Model buttons
        models = [
            ("SPC-CNN", self.runSPCCNN, "#D35B58"),
            ("Naive Bayes", self.runNaiveBayes, "#5D9B9A"),
            ("SVM", self.runSVM, "#8A6EB8")
        ]
        
        for text, command, color in models:
            btn = ctk.CTkButton(left_panel, text=text, command=command,
                               fg_color=color, hover_color=self.darken_color(color),
                               height=40, corner_radius=6)
            btn.pack(fill="x", pady=5)
        
        # Model info panel
        info_frame = ctk.CTkFrame(left_panel)
        info_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(info_frame, text="Model Info", font=("Arial", 14, "bold")).pack()
        
        self.model_info = {
            "status": ctk.CTkLabel(info_frame, text="Status: Not trained", anchor="w"),
            "accuracy": ctk.CTkLabel(info_frame, text="Accuracy: -", anchor="w"),
            "best_model": ctk.CTkLabel(info_frame, text="Best: -", anchor="w")
        }
        
        for label in self.model_info.values():
            label.pack(fill="x", pady=2)
        
        # Right panel - output and visualizations
        right_panel = ctk.CTkFrame(content)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Notebook for output and plots
        self.model_notebook = ctk.CTkTabview(right_panel)
        self.model_notebook.pack(fill="both", expand=True)
        
        # Console output tab
        self.model_console = self.model_notebook.add("Console")
        self.model_text = ctk.CTkTextbox(self.model_console, wrap="word", font=("Consolas", 12))
        self.model_text.pack(fill="both", expand=True)
        
        # Visualization tab
        self.model_viz = self.model_notebook.add("Training Metrics")
        self.model_viz_frame = ctk.CTkFrame(self.model_viz, fg_color="transparent")
        self.model_viz_frame.pack(fill="both", expand=True)
        
        # Initially empty visualization area
        self.model_viz_label = ctk.CTkLabel(self.model_viz_frame, text="Training metrics will appear here",
                                          font=("Arial", 14))
        self.model_viz_label.pack(pady=50)
        
        return tab
    
    def create_results_tab(self):
        """Create the results and testing tab"""
        tab = ctk.CTkFrame(self.main_content, fg_color="transparent")
        
        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header, text="Results & Testing", font=("Arial", 20, "bold")).pack(side="left")
        
        # Main content area
        content = ctk.CTkFrame(tab)
        content.pack(fill="both", expand=True)
        
        # Left panel - buttons and info
        left_panel = ctk.CTkFrame(content, width=250)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Action buttons
        actions = [
            ("Comparison Graph", self.graph, "#5D9B9A"),
            ("Predict Attack", self.predict, "#E1A730"),
           # ("Export Results", self.export_results, "#8A6EB8")
        ]
        
        for text, command, color in actions:
            btn = ctk.CTkButton(left_panel, text=text, command=command,
                               fg_color=color, hover_color=self.darken_color(color),
                               height=40, corner_radius=6)
            btn.pack(fill="x", pady=5)
        
        # Results info panel
        info_frame = ctk.CTkFrame(left_panel)
        info_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(info_frame, text="Results Summary", font=("Arial", 14, "bold")).pack()
        
        self.results_info = {
            "best_model": ctk.CTkLabel(info_frame, text="Best Model: -", anchor="w"),
            "best_accuracy": ctk.CTkLabel(info_frame, text="Best Accuracy: -", anchor="w"),
            "test_samples": ctk.CTkLabel(info_frame, text="Test Samples: 0", anchor="w")
        }
        
        for label in self.results_info.values():
            label.pack(fill="x", pady=2)
        
        # Right panel - output and visualizations
        right_panel = ctk.CTkFrame(content)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Notebook for output and plots
        self.results_notebook = ctk.CTkTabview(right_panel)
        self.results_notebook.pack(fill="both", expand=True)
        
        # Console output tab
        self.results_console = self.results_notebook.add("Console")
        self.results_text = ctk.CTkTextbox(self.results_console, wrap="word", font=("Consolas", 12))
        self.results_text.pack(fill="both", expand=True)
        
        # Visualization tab
        self.results_viz = self.results_notebook.add("Results Visualization")
        self.results_viz_frame = ctk.CTkFrame(self.results_viz, fg_color="transparent")
        self.results_viz_frame.pack(fill="both", expand=True)
        
        # Initially empty visualization area
        self.results_viz_label = ctk.CTkLabel(self.results_viz_frame, text="Results visualizations will appear here",
                                            font=("Arial", 14))
        self.results_viz_label.pack(pady=50)
        
        return tab
    
    def create_dashboard_tab(self):
        """Create the system dashboard tab"""
        tab = ctk.CTkFrame(self.main_content, fg_color="transparent")
        
        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header, text="System Dashboard", font=("Arial", 20, "bold")).pack(side="left")
        
        # Main content area
        content = ctk.CTkFrame(tab)
        content.pack(fill="both", expand=True)
        
        # Create metrics cards
        metrics_frame = ctk.CTkFrame(content)
        metrics_frame.pack(fill="x", pady=10)
        
        self.metrics_cards = {
            "data": self.create_metric_card(metrics_frame, "Data Quality", "0%", "#5D9B9A"),
            "model": self.create_metric_card(metrics_frame, "Model Accuracy", "0%", "#8A6EB8"),
            "performance": self.create_metric_card(metrics_frame, "System Performance", "0%", "#D35B58"),
            "threats": self.create_metric_card(metrics_frame, "Threats Detected", "0", "#E1A730")
        }
        
        # Create visualization area
        viz_frame = ctk.CTkFrame(content)
        viz_frame.pack(fill="both", expand=True)
        
        # Placeholder for dashboard visualizations
        self.dashboard_viz_label = ctk.CTkLabel(viz_frame, 
                                               text="System performance metrics and visualizations will appear here",
                                               font=("Arial", 14))
        self.dashboard_viz_label.pack(pady=50)
        
        return tab
    
    def create_metric_card(self, parent, title, value, color):
        """Create a metric card for the dashboard"""
        card = ctk.CTkFrame(parent, width=200, height=120, fg_color=color, corner_radius=10)
        card.pack_propagate(False)
        card.pack(side="left", padx=10, pady=5)
        
        ctk.CTkLabel(card, text=title, font=("Arial", 14, "bold")).pack(pady=(15, 5))
        ctk.CTkLabel(card, text=value, font=("Arial", 24, "bold")).pack(pady=5)
        
        return card
    
    def darken_color(self, hex_color, factor=0.8):
        """Darken a hex color by a factor"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, int(c * factor)) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
    
    def show_tab(self, tab_name):
        """Show the selected tab"""
        self.current_tab = tab_name
        for name, tab in self.tabs.items():
            if name == tab_name:
                tab.pack(fill="both", expand=True)
            else:
                tab.pack_forget()
    
    def show_tutorial(self):
        """Show tutorial popup"""
        tutorial = ctk.CTkToplevel(self)
        tutorial.title("Quick Tutorial")
        tutorial.geometry("800x600")
        tutorial.grab_set()
        
        # Notebook for tutorial sections
        notebook = ctk.CTkTabview(tutorial)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tutorial tabs
        tabs = [
            ("Getting Started", "Welcome !;\n\nThis system helps detect network threats using advanced ML models."),
            ("Data Upload", "1. Click 'Upload Dataset' to load your network data\n2. Supported formats: CSV, JSON\n3. Data should include network features and labels"),
            ("Preprocessing", "1. Click 'Preprocess' to clean and normalize data\n2. The system handles missing values and encoding\n3. Visualizations show data distribution"),
            ("Model Training", "1. Choose between SPC-CNN, Naive Bayes, or SVM\n2. Training progress is shown in real-time\n3. Metrics are calculated automatically")
        ]
        
        for name, text in tabs:
            tab = notebook.add(name)
            content = ctk.CTkFrame(tab, fg_color="transparent")
            content.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Text content
            ctk.CTkLabel(content, text=text, font=("Arial", 14), 
                        justify="left", wraplength=700).pack(anchor="w", pady=5)
            
            # Add example image if available
            if name.lower() in self.tutorial_images:
                img_label = ctk.CTkLabel(content, image=self.tutorial_images[name.lower()], text="")
                img_label.pack(pady=10)
    
    def update_data_info(self):
        """Update the data info panel"""
        if self.dataset is not None:
            self.data_info["samples"].configure(text=f"Samples: {len(self.dataset)}")
            self.data_info["features"].configure(text=f"Features: {len(self.dataset.columns)-1}")
            self.data_info["classes"].configure(text=f"Classes: {len(np.unique(self.dataset['label'])) if 'label' in self.dataset.columns else 'Unknown'}")
            self.data_info["status"].configure(text="Status: Ready", text_color="green")
            self.status_labels["data"].configure(text="Data: Loaded", text_color="green")
        else:
            self.data_info["samples"].configure(text="Samples: 0")
            self.data_info["features"].configure(text="Features: 0")
            self.data_info["classes"].configure(text="Classes: 0")
            self.data_info["status"].configure(text="Status: Not loaded", text_color="red")
    
    def update_model_info(self, model_name, accuracy):
        """Update the model info panel"""
        self.model_info["status"].configure(text=f"Status: {model_name} trained", text_color="green")
        self.model_info["accuracy"].configure(text=f"Accuracy: {accuracy:.2f}%")
    
        # Update best model if this is better
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.model_info["best_model"].configure(text=f"Best: {model_name} ({accuracy:.2f}%)")
    
        self.status_labels["model"].configure(text="Model: Trained", text_color="green")
    
    def update_results_info(self):
        """Update the results info panel"""
        if len(self.accuracy) > 0:
            best_idx = np.argmax(self.accuracy)
            models = ["SPC-CNN", "Naive Bayes", "SVM"]
            self.results_info["best_model"].configure(text=f"Best Model: {models[best_idx]}")
            self.results_info["best_accuracy"].configure(text=f"Best Accuracy: {self.accuracy[best_idx]:.2f}%")
        
        if self.X_test is not None:
            self.results_info["test_samples"].configure(text=f"Test Samples: {len(self.X_test)}")
    
    def update_dashboard(self):
        """Update the dashboard metrics"""
        if self.dataset is not None:
            quality = min(100, 90 + np.random.randint(0, 10))  # Simulated data quality metric
            self.metrics_cards["data"].children["!ctklabel2"].configure(text=f"{quality}%")
    
        if len(self.accuracy) > 0:
            best_acc = max(self.accuracy)
            self.metrics_cards["model"].children["!ctklabel2"].configure(text=f"{best_acc:.1f}%")
            threats_detected = int(best_acc * len(self.X_test)/100 if self.X_test is not None else 0)
            self.metrics_cards["threats"].children["!ctklabel2"].configure(text=str(threats_detected))
    
        # Update performance metric card with the stored best accuracy
        self.metrics_cards["performance"].children["!ctklabel2"].configure(text=f"{self.best_accuracy:.1f}%")
    
    def log_message(self, message, tab=None):
        """Log messages to appropriate tab's text widget"""
        if tab is None:
            tab = self.current_tab
        
        text_widget = {
            "data": self.data_text,
            "model": self.model_text,
            "results": self.results_text
        }.get(tab, self.data_text)
        
        text_widget.insert("end", message + "\n")
        text_widget.see("end")
        self.update_idletasks()
    
    def show_plot_in_tab(self, fig, tab_name):
        """Display a matplotlib figure in the specified tab"""
        # Clear previous content
        viz_frame = {
            "data": self.data_viz_frame,
            "model": self.model_viz_frame,
            "results": self.results_viz_frame
        }.get(tab_name)
        
        for widget in viz_frame.winfo_children():
            widget.destroy()
        
        # Create canvas and display plot
        canvas = FigureCanvasTkAgg(fig, master=viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar would require NavigationToolbar2Tk which isn't directly compatible
        # with customtkinter, so we'll skip it for now
    
    # Original functionality methods with UI enhancements
    def uploadDataset(self): 
        filename = filedialog.askopenfilename(initialdir="Dataset",
                                            filetypes=[("CSV Files", "*.csv"), 
                                                       ("All Files", "*.*")])
        if filename:
            self.log_message(f"Loading dataset from {filename}")
            try:
                self.dataset = pd.read_csv(filename)
                self.log_message("Dataset loaded successfully:\n")
                self.log_message(str(self.dataset.head()))
                
                # Update UI
                self.update_data_info()
                self.update_dashboard()
                
                # Show data distribution
                if 'label' in self.dataset.columns:
                    self.labels = np.unique(self.dataset['label'])
                    label_counts = self.dataset.groupby('label').size()
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    label_counts.plot(kind="bar", ax=ax, color='#3A7EBF')
                    ax.set_title("Attack Type Distribution", pad=20)
                    ax.set_xlabel("Attack Type")
                    ax.set_ylabel("Count")
                    fig.tight_layout()
                    
                    self.show_plot_in_tab(fig, "data")
                    
            except Exception as e:
                self.log_message(f"Error loading dataset: {str(e)}")
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def preprocess(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return
        
        self.data_text.delete("1.0", "end")
        cols = ['protocol_type','service','flag','label']
        
        try:
            self.log_message("Starting preprocessing...")
            
            # Handle missing values
            self.dataset.fillna(0, inplace=True)
            
            # Encode categorical features
            self.dataset[cols[0]] = pd.Series(self.le1.fit_transform(self.dataset[cols[0]].astype(str)))
            self.dataset[cols[1]] = pd.Series(self.le2.fit_transform(self.dataset[cols[1]].astype(str)))
            self.dataset[cols[2]] = pd.Series(self.le3.fit_transform(self.dataset[cols[2]].astype(str)))
            self.dataset[cols[3]] = pd.Series(self.le4.fit_transform(self.dataset[cols[3]].astype(str)))
            
            # Prepare X and Y
            data = self.dataset.values
            self.X = data[:,0:data.shape[1]-1]
            self.Y = data[:,data.shape[1]-1]
            
            # Shuffle data
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.Y = self.Y[indices]
            
            self.log_message("Preprocessing completed successfully!\n")
            self.log_message("Sample of processed data:\n")
            self.log_message(str(self.dataset.head()))
            
            # Update UI
            self.update_data_info()
            self.update_dashboard()
            
            # Show feature distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            pd.DataFrame(self.X).iloc[:,:10].hist(ax=ax, color='#3A7EBF')
            fig.tight_layout()
            self.show_plot_in_tab(fig, "data")
            
        except Exception as e:
            self.log_message(f"Error during preprocessing: {str(e)}")
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
    
    def augmentation(self):
        if self.X is None or self.Y is None:
            messagebox.showwarning("Warning", "Please preprocess data first")
            return
        
        self.data_text.delete("1.0", "end")
        
        try:
            self.log_message("Applying ADASYN augmentation...")
            
            sm = SMOTE(random_state=2)
            self.X, self.Y = sm.fit_resample(self.X, self.Y)
            
            unique, count = np.unique(self.Y, return_counts=True)
            self.log_message("\nRecords in each class after augmentation:\n")
            for i, lbl in enumerate(self.labels):
                self.log_message(f"{lbl}: {count[i]} records")
            
            # Update UI
            self.update_data_info()
            self.update_dashboard()
            
            # Show augmented distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(self.labels, count, color='#5D9B9A')
            ax.set_title("Class Distribution After ADASYN", pad=20)
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            fig.tight_layout()
            self.show_plot_in_tab(fig, "data")
            
        except Exception as e:
            self.log_message(f"Error during augmentation: {str(e)}")
            messagebox.showerror("Error", f"Augmentation failed: {str(e)}")
    
    def trainTestSplit(self):
        if self.X is None or self.Y is None:
            messagebox.showwarning("Warning", "Please preprocess data first")
            return
        
        self.data_text.delete("1.0", "end")
        
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2)
            
            self.log_message("Dataset split into training and test sets:\n")
            self.log_message(f"Total records: {self.X.shape[0]}")
            self.log_message(f"Training set: {self.X_train.shape[0]} records (80%)")
            self.log_message(f"Test set: {self.X_test.shape[0]} records (20%)")
            
            # Update UI
            self.update_data_info()
            self.update_dashboard()
            self.update_results_info()
            
            # Show split visualization
            fig, ax = plt.subplots(figsize=(6, 6))
            sizes = [len(self.X_train), len(self.X_test)]
            labels = ['Training Set', 'Test Set']
            colors = ['#3A7EBF', '#E1A730']
            explode = (0.1, 0)
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                  autopct='%1.1f%%', shadow=True, startangle=140)
            ax.axis('equal')
            ax.set_title("Train-Test Split")
            
            fig.tight_layout()
            self.show_plot_in_tab(fig, "data")
            
        except Exception as e:
            self.log_message(f"Error during train-test split: {str(e)}")
            messagebox.showerror("Error", f"Train-test split failed: {str(e)}")
    
    def calculateMetrics(self, algorithm, predict, y_test):
        a = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100
        
        self.accuracy.append(a)
        self.precision.append(p)
        self.recall.append(r)
        self.fscore.append(f)
        
        self.log_message(f"{algorithm} Performance Metrics:", "model")
        self.log_message(f"Accuracy: {a:.2f}%", "model")
        self.log_message(f"Precision: {p:.2f}%", "model")
        self.log_message(f"Recall: {r:.2f}%", "model")
        self.log_message(f"F1 Score: {f:.2f}%\n", "model")
        
        # Update model info
        self.update_model_info(algorithm, a)
        self.update_dashboard()
        self.update_results_info()
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, predict)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.labels, yticklabels=self.labels, ax=ax)
        ax.set_title(f"{algorithm} Confusion Matrix")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        fig.tight_layout()
        self.show_plot_in_tab(fig, "model")
    
    def runSPCCNN(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Warning", "Please split data first")
            return
        
        self.model_text.delete("1.0", "end")
        self.log_message("Starting SPC-CNN training...", "model")
        
        try:
            # Reshape data for CNN
            X_train1 = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1, 1))
            X_test1 = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1, 1))
            y_train1 = to_categorical(self.y_train)
            y_test1 = to_categorical(self.y_test)
            
            # Create model
            self.spc_cnn = Sequential()
            self.spc_cnn.add(Convolution2D(32, (1, 1), 
                              input_shape=(X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), 
                              activation='relu'))
            self.spc_cnn.add(MaxPooling2D(pool_size=(1, 1)))
            self.spc_cnn.add(Convolution2D(32, (1, 1), activation='relu'))
            self.spc_cnn.add(MaxPooling2D(pool_size=(1, 1)))
            self.spc_cnn.add(Flatten())
            self.spc_cnn.add(Dense(256, activation='relu'))
            self.spc_cnn.add(Dense(y_train1.shape[1], activation='softmax'))
            
            self.spc_cnn.compile(optimizer='adam', 
                               loss='categorical_crossentropy', 
                               metrics=['accuracy'])
            
            # Check for existing model
            os.makedirs('model', exist_ok=True)
            model_path = 'model/model_weights.hdf5'
            
            if os.path.exists(model_path):
                self.log_message("Loading existing model weights...", "model")
                self.spc_cnn.load_weights(model_path)
            else:
                self.log_message("Training new model...", "model")
                checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
                
                history = self.spc_cnn.fit(X_train1, y_train1, 
                                         batch_size=16, 
                                         epochs=20,
                                         validation_data=(X_test1, y_test1),
                                         callbacks=[checkpoint],
                                         verbose=1)
                
                # Save training history
                with open('model/history.pckl', 'wb') as f:
                    pickle.dump(history.history, f)
                
                # Plot training history
                fig, ax = plt.subplots(2, 1, figsize=(8, 8))
                
                # Accuracy plot
                ax[0].plot(history.history['accuracy'], label='Train Accuracy', color='#3A7EBF')
                ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#E1A730')
                ax[0].set_title('Model Accuracy')
                ax[0].set_ylabel('Accuracy')
                ax[0].legend(loc='lower right')
                
                # Loss plot
                ax[1].plot(history.history['loss'], label='Train Loss', color='#3A7EBF')
                ax[1].plot(history.history['val_loss'], label='Validation Loss', color='#E1A730')
                ax[1].set_title('Model Loss')
                ax[1].set_ylabel('Loss')
                ax[1].set_xlabel('Epoch')
                ax[1].legend(loc='upper right')
                
                fig.tight_layout()
                self.show_plot_in_tab(fig, "model")
            
            # Evaluate model
            predict = self.spc_cnn.predict(X_test1)
            predict = np.argmax(predict, axis=1)
            target = np.argmax(y_test1, axis=1)
            
            self.calculateMetrics("SPC-CNN", predict, target)
            
        except Exception as e:
            self.log_message(f"Error during SPC-CNN training: {str(e)}", "model")
            messagebox.showerror("Error", f"SPC-CNN training failed: {str(e)}")
    
    def runNaiveBayes(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Warning", "Please split data first")
            return
        
        self.model_text.delete("1.0", "end")
        self.log_message("Training Naive Bayes model...", "model")
        
        try:
            nb = GaussianNB()
            nb.fit(self.X_train, self.y_train)
            predict = nb.predict(self.X_test)
            
            self.calculateMetrics("Naive Bayes", predict, self.y_test)
            
        except Exception as e:
            self.log_message(f"Error during Naive Bayes training: {str(e)}", "model")
            messagebox.showerror("Error", f"Naive Bayes training failed: {str(e)}")
    
    def runSVM(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Warning", "Please split data first")
            return
        
        self.model_text.delete("1.0", "end")
        self.log_message("Training SVM model...", "model")
        
        try:
            svm_cls = svm.SVC()
            svm_cls.fit(self.X_train, self.y_train)
            predict = svm_cls.predict(self.X_test)
            
            self.calculateMetrics("SVM", predict, self.y_test)
            
        except Exception as e:
            self.log_message(f"Error during SVM training: {str(e)}", "model")
            messagebox.showerror("Error", f"SVM training failed: {str(e)}")
    
    def graph(self):
        if len(self.accuracy) == 0:
            messagebox.showwarning("Warning", "Please train at least one model first")
            return
        
        self.results_text.delete("1.0", "end")
        self.log_message("Generating comparison graphs...", "results")
        
        try:
            # Create HTML comparison table
            output = """<html><head><style>
                body {font-family: Arial, sans-serif; margin: 20px;}
                table {border-collapse: collapse; width: 80%; margin: 20px auto;}
                th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
                th {background-color: #3A7EBF; color: white;}
                tr:nth-child(even) {background-color: #f2f2f2;}
                .best {font-weight: bold; color: #D35B58;}
            </style></head><body>"""
            
            output += "<h2 style='text-align:center'>Model Performance Comparison</h2>"
            output += "<table align='center'><tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>"
            
            models = ["SPC-CNN", "Naive Bayes", "SVM"]
            best_acc_idx = np.argmax(self.accuracy)
            best_pre_idx = np.argmax(self.precision)
            best_rec_idx = np.argmax(self.recall)
            best_f1_idx = np.argmax(self.fscore)
            
            for i, model in enumerate(models):
                acc_class = "best" if i == best_acc_idx else ""
                pre_class = "best" if i == best_pre_idx else ""
                rec_class = "best" if i == best_rec_idx else ""
                f1_class = "best" if i == best_f1_idx else ""
                
                output += f"<tr><td>{model}</td>"
                output += f"<td class='{acc_class}'>{self.accuracy[i]:.2f}%</td>"
                output += f"<td class='{pre_class}'>{self.precision[i]:.2f}%</td>"
                output += f"<td class='{rec_class}'>{self.recall[i]:.2f}%</td>"
                output += f"<td class='{f1_class}'>{self.fscore[i]:.2f}%</td></tr>"
            
            output += "</table></body></html>"
            
            with open("comparison.html", "w") as f:
                f.write(output)
            
            webbrowser.open("comparison.html", new=2)
            
            # Create bar chart comparison
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [self.accuracy, self.precision, self.recall, self.fscore]
            colors = ['#3A7EBF', '#5D9B9A', '#8A6EB8', '#E1A730']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.2
            index = np.arange(len(metrics))
            
            for i, model in enumerate(models):
                ax.bar(index + i*bar_width, [values[j][i] for j in range(len(metrics))], 
                      bar_width, label=model, color=colors[i])
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score (%)')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(index + bar_width)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_ylim(0, 110)
            
            fig.tight_layout()
            self.show_plot_in_tab(fig, "results")
            
        except Exception as e:
            self.log_message(f"Error generating graphs: {str(e)}", "results")
            messagebox.showerror("Error", f"Graph generation failed: {str(e)}")
    
    def predict(self):
        if self.spc_cnn is None:
            messagebox.showwarning("Warning", "Please train the SPC-CNN model first")
            return
        
        cols = ['protocol_type','service','flag']
        self.results_text.delete("1.0", "end")
        
        filename = filedialog.askopenfilename(initialdir="Dataset",
                                            filetypes=[("CSV Files", "*.csv"), 
                                                       ("All Files", "*.*")])
        if not filename:
            return
        
        try:
            self.log_message(f"Loading test data from {filename}", "results")
            dataset = pd.read_csv(filename)
            dataset.fillna(0, inplace=True)
            
            # Save original data for display
            original_data = dataset.copy()
            
            # Preprocess the test data
            dataset[cols[0]] = pd.Series(self.le1.transform(dataset[cols[0]].astype(str)))
            dataset[cols[1]] = pd.Series(self.le2.transform(dataset[cols[1]].astype(str)))
            dataset[cols[2]] = pd.Series(self.le3.transform(dataset[cols[2]].astype(str)))
            
            # Prepare for prediction
            test_data = dataset.values
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1, 1))
            
            # Make predictions
            predictions = self.spc_cnn.predict(test_data)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Display results
            self.log_message("\nPrediction Results:\n", "results")
            self.log_message("{:<10} {:<20} {:<10} {:<10}".format(
                "Sample", "Protocol", "Service", "Prediction"), "results")
            self.log_message("-"*50, "results")
            
            for i in range(min(20, len(predicted_classes))):  # Show first 20 results
                sample = original_data.iloc[i]
                self.log_message("{:<10} {:<20} {:<10} {:<10}".format(
                    i+1, 
                    sample['protocol_type'], 
                    sample['service'], 
                    self.labels[predicted_classes[i]]), "results")
            
            # Show prediction distribution
            unique, counts = np.unique(predicted_classes, return_counts=True)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar([self.labels[u] for u in unique], counts, color='#D35B58')
            ax.set_title("Predicted Attack Distribution", pad=20)
            ax.set_xlabel("Attack Type")
            ax.set_ylabel("Count")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            fig.tight_layout()
            self.show_plot_in_tab(fig, "results")
            
            # Update dashboard
            self.update_dashboard()
            
        except Exception as e:
            self.log_message(f"Prediction error: {str(e)}", "results")
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def export_results(self):
        if len(self.accuracy) == 0:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                   filetypes=[("Excel files", "*.xlsx"),
                                                              ("All files", "*.*")])
            if not filename:
                return
            
            # Create a DataFrame with results
            results_df = pd.DataFrame({
                'Model': ['SPC-CNN', 'Naive Bayes', 'SVM'],
                'Accuracy': self.accuracy,
                'Precision': self.precision,
                'Recall': self.recall,
                'F1 Score': self.fscore
            })
            
            # Save to Excel
            results_df.to_excel(filename, index=False)
            self.log_message(f"Results exported to {filename}", "results")
            messagebox.showinfo("Success", "Results exported successfully")
            
        except Exception as e:
            self.log_message(f"Export error: {str(e)}", "results")
            messagebox.showerror("Error", f"Export failed: {str(e)}")

if __name__ == "__main__":
    app = IDS()
    app.mainloop()