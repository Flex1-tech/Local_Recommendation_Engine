from PIL import Image
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
import fleep
from mutagen import File
import subprocess




def get_media_type(filepath):
    """
    Retourne: 'audio', 'video' ou None
    """
    try:
        with open(filepath, "rb") as file:
            info = fleep.get(file.read(128))

        types = info.type

        if not types:
            return None

        if "audio" in types:
            return "audio"
        elif "video" in types:
            return "video"

        return None

    except:
        return None

def is_valid_media(filepath):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except:
        return False

def is_audio_file(filepath):
    with open(filepath, "rb") as file:
        info = fleep.get(file.read(128))
    return "audio" in info.type or "video" in info.type  # Some audio files may be classified as video

def open_file(root, valid_files=None):
    if valid_files is None:
        valid_files = {}
    
    try:
        file_paths = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[("All files", "*.*")]
        )
        
        invalid_count = 0

        for path in file_paths:
            if path in valid_files:
                continue  # éviter doublons

            if is_audio_file(path) and is_valid_media(path):
                valid_files[path] = False
            else:
                invalid_count += 1
        

        if invalid_count > 0:
            show_toast(root, f"{invalid_count} fichiers ignorés")
        elif len(valid_files) == 0:
            show_toast(root, "Aucun fichier audio valide sélectionné")
        
        return valid_files

    except Exception:
        show_toast(root, "Erreur lors de l'ouverture des fichiers")
        return {}


def show_toast(root, message, duration=3000):

    error_icon = ctk.CTkImage(Image.open("assets/error.png"), size=(20, 20))

    toast = ctk.CTkToplevel(root)
    toast.overrideredirect(True)
    toast.attributes("-topmost", True)

    frame = ctk.CTkFrame(toast, corner_radius=10,border_width=2, border_color="#F36C19")
    frame.pack(padx=0, pady=0, fill="both", expand=True)
    bg = root.cget("fg_color")

    if isinstance(bg, tuple):
        bg = bg[0]  # mode light/dark safe

    toast.configure(fg_color=bg)
    frame.configure(fg_color=bg)

    label = ctk.CTkLabel(
        frame,
        text=message,
        text_color="#F36C19",
        image=error_icon,
        compound="left",
        font=("Arial", 12)
    )
    label.pack(padx=15, pady=10)

    root.update()  
    toast.update_idletasks()

    x = root.winfo_rootx() + root.winfo_width() - toast.winfo_width() - 20
    y = root.winfo_rooty() + root.winfo_height() - toast.winfo_height() - 20

    toast.geometry(f"{toast.winfo_width()}x{toast.winfo_height()}+{x}+{y}")

    toast.after(duration, toast.destroy)

