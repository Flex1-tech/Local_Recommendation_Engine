from functools import partial
import subprocess
import sys

from PIL import Image
import customtkinter as ctk
from func import open_file, show_toast
from extraction import recommend_playlist, initialize_database, load_cnn14, make_m3u
from pathlib import Path
import threading 

upload_icon = ctk.CTkImage(light_image=Image.open("assets/light_upload.png"),dark_image=Image.open("assets/dark_upload.png"), size=(30, 30))
start_icon = ctk.CTkImage(light_image=Image.open("assets/start.png"), dark_image=Image.open("assets/start_light.png"), size=(30, 30))
remove_icon = ctk.CTkImage(Image.open('assets/streamline--delete-1-remix.png'),size=(20,20))
heart_off = ctk.CTkImage(Image.open("assets/coeur_gris.png"), size=(30, 30))
heart_on = ctk.CTkImage(Image.open("assets/coeur_rouge.png"), size=(30, 30))

# customtkinter.set_appearance_mode("System")  # "Light" ou "Dark"
# customtkinter.set_default_color_theme("green") 76


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.selected_files = {}
        self.file_frame = None
        self.start_button = None
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.loader_base = Image.open("assets/loader.png")
        self.loader_frames = [
            ctk.CTkImage(self.loader_base.rotate(angle), size=(40, 40))
            for angle in range(0, 360, 15)
        ]
        self.loader_index = 0
        self.loading = False
        self.file_widgets = {}

        width = int(screen_width * 0.7)
        height = int(screen_height * 0.7)

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.geometry(f"{width}x{height}+{x}+{y}")
        self.title("Musical Recommender System")
        self.configure(border_color=("black", "white"))
        
        # Main Frame 
        self.main_frame = ctk.CTkFrame(
            self,
            border_width=2,
            corner_radius=10
        )
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.main_frame.grid_rowconfigure(0, weight=0)  # top_bar
        self.main_frame.grid_rowconfigure(1, weight=0)  # header
        self.main_frame.grid_rowconfigure(2, weight=1)  # file_frame (prend l’espace)
        self.main_frame.grid_rowconfigure(3, weight=0)  # bottom_bar

        # Frame pour les bouton du bas
        self.bottom_bar = ctk.CTkFrame(
            self.main_frame, 
            fg_color="transparent"
        )
        self.bottom_bar_inner = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.bottom_bar_inner.pack(expand=True)
        self.import_button = ctk.CTkButton(
            self.bottom_bar_inner,
            text="Importer plus!",
            fg_color="#FF8E25",
            hover_color="#F36C19",
            image=upload_icon,
            compound="left",
            font=("Arial", 18),
            command=self.import_files
        )
        self.start_button = ctk.CTkButton(
            self.bottom_bar_inner, 
            text="Commencez..!", 
            font=("Arial", 18), 
            state="disabled", 
            fg_color="#FF8E25", 
            hover_color="#F36C19",
            image=start_icon,
            compound="left",
            command=self.start
        )



        # Barre de recherche
        self.search_visible = False
        self.top_bar = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.search_entry = ctk.CTkEntry(self.top_bar, placeholder_text="Rechercher une musique...", width=250, font=("Arial", 14))
        self.search_entry.bind("<KeyRelease>", lambda event: self.refresh_file_list())
        self.search_entry.pack(side="left", padx=10)
        
        # Reinitialisation
        self.reset_button = ctk.CTkButton(self.top_bar, text="Réinitialiser", font=("Arial", 14), command=self.reset, fg_color="#FF8E25", hover_color="#F36C19")
        self.reset_button.pack(side="right", padx=10)

        # Button Start

        self.label = ctk.CTkLabel(
            self.main_frame,
            text="Écoutez. Likez. Découvrez.\nVotre musique, parfaitement orchestrée.",
            font=("Arial", 19, "bold"),
            justify="center",
            anchor="center"
        )
        self.label.grid(row=1, column=0, pady=(50,1), sticky="ew")

        self.loader_label = ctk.CTkLabel(self.main_frame, text="")
        self.loader_text = ctk.CTkLabel(self.main_frame, text="Chargement...", font=("Arial", 16, "bold"))

        self.center_button = ctk.CTkButton(
            self.main_frame,
            text="Importer mes musiques",
            fg_color="#FF8E25",
            hover_color="#F36C19",
            image=upload_icon,
            compound="left",
            font=("Arial", 17),
            command=self.import_files,
        )
        self.center_button.grid(row=2, column=0)

        # événements
        self.search_entry.bind("<KeyRelease>", lambda e: self.refresh_file_list())
        self.search_entry.bind("<FocusIn>", lambda e: self.search_entry.configure(border_color="#F36C19"))
        self.search_entry.bind("<FocusOut>", lambda e: self.search_entry.configure(border_color="#444"))
        # Sur chaque widget cliquable, retirer le focus de la search bar
        self.bind_all("<Button-1>", lambda e: self.focus_set() if not str(e.widget).startswith(str(self.search_entry)) else None)

    def import_files(self):
        # afficher loader
        self.show_loader()
        thread = threading.Thread(target=self.load_files, daemon=True)
        thread.start()

    def get_selected_files(self):
        return self.selected_files
    
    def clear_selected_files(self):
        self.selected_files = {}

    def remove_file(self, file):
        if file in self.selected_files:
            del self.selected_files[file]

            # supprimer seulement le frame
            if file in self.file_widgets:
                self.file_widgets[file].destroy()
                del self.file_widgets[file]

            self.refresh_label()
        self.update_start_button()

    def refresh_label(self):
        count = len(self.selected_files)

        if count == 0:
            self.hide_search_bar()
            if self.file_frame:
                self.file_frame.grid_forget()
                self.file_frame.destroy()
                self.file_frame = None
            
            self.label.configure(
                text="Écoutez. Likez. Découvrez.\nVotre musique, parfaitement orchestrée."
            )
            self.center_button.grid(row=2, column=0)
            self.top_bar.grid_forget()

        else:
            self.label.configure(text=f"{count} fichier(s) importé(s)")
        
        self.update_idletasks()

    def reset(self):
        # 1. vider les données
        self.clear_selected_files()

        # 2. supprimer tous les widgets de fichiers
        for frame in self.file_widgets.values():
            frame.destroy()
        self.file_widgets.clear()

        # 3. supprimer le file_frame
        if self.file_frame:
            self.file_frame.grid_forget()
            self.file_frame.destroy()
            self.file_frame = None
        

        # 4. cacher la barre de recherche et la barre du bas
        self.bottom_bar.grid_forget()
        self.start_button.pack_forget()
        self.import_button.pack_forget()
        self.hide_search_bar()

        # 5. vider la recherche
        self.search_entry.delete(0, "end")

        # 6. reset du label
        self.label.configure(
            text="Écoutez. Likez. Découvrez.\nVotre musique, parfaitement orchestrée."
        )

        # 7. remettre le bouton au centre
        self.center_button.grid(row=2, column=0)
        self.main_frame.update_idletasks()
        self.update_idletasks()

    def toggle_like(self, file, like_btn):
        self.selected_files[file] = not self.selected_files[file]

        # Changer l'image
        like_btn.configure(
            image=heart_on if self.selected_files[file] else heart_off
        )
        self.update_start_button()

    def get_heart_icon(self, file):
        return heart_on if self.selected_files[file] else heart_off

    def load_files(self):
        try:
            files = open_file(self, self.selected_files)
        except Exception as e:
            self.after(0, lambda: self.hide_loader())  # cacher le loader
            show_toast(self, "Erreur lors de l'ouverture des fichiers")
        
        if len(files) <=3:
            self.after(0, lambda: show_toast(self, "Importez au moins 4 fichiers\n pour de meilleures recommandations!"))
        if not files:
            self.after(0, self.on_files_cancelled)  # cacher le loader
        else:
            self.after(0, lambda: self.on_files_loaded(files))

    def show_loader(self):
        self.loading = True
        if len(self.selected_files) == 0:
            self.center_button.grid_forget()
        else:
            self.import_button.configure(state="disabled")

        self.loader_label.place(relx=0.5, rely=0.45, anchor="center")
        self.loader_text.place(relx=0.5, rely=0.55, anchor="center")
        self.rotate_loader()

    def hide_loader(self):
        self.loading = False
        self.loader_label.place_forget()
        self.loader_text.place_forget()

    def rotate_loader(self):
        if not self.loading:
            return

        img = self.loader_frames[self.loader_index]
        self.loader_label.configure(image=img)
        self.loader_label.image = img

        self.loader_index = (self.loader_index + 1) % len(self.loader_frames)

        self.after(50, self.rotate_loader)

    def on_files_loaded(self, files):
        if not files:
            self.hide_loader()
            return
        self.selected_files.update(files)
        self.hide_loader()

        if not self.search_visible:
            self.top_bar.grid(row=0, column=0, sticky="ew", pady=10)
            self.search_visible = True
        
        self.ensure_file_frame()
        self.add_new_files_to_ui(files)
        self.refresh_file_list()
        self.update_import_button()
        self.update_start_button()

    def ensure_file_frame(self):
        if not self.file_frame or not self.file_frame.winfo_exists():
            self.file_frame = ctk.CTkScrollableFrame(
                self.main_frame,
                label_text="Vos fichiers",
                border_width=2,
                corner_radius=10,
                width=600,
                height=230
            )
            self.center_button.grid_forget()
            self.file_frame.grid(row=2, column=0, sticky="nsew", padx=100, pady=10)

    def update_import_button(self):
        # Afficher le bottom bar si pas déjà fait
        if not self.bottom_bar.winfo_ismapped():
            self.bottom_bar.grid(row=3, column=0, sticky="ew", pady=20)

        if not self.import_button.winfo_ismapped():
            self.import_button.pack(side="left", padx=10)
    
        self.import_button.configure(state="normal")

    def on_files_cancelled(self):
        self.hide_loader()
        if len(self.selected_files) == 0:
        # remettre le bouton
            self.center_button.grid(row=2, column=0)
            self.center_button.configure(state="normal")
        else:
            self.update_import_button()
            self.import_button.configure(state="normal")

    def build_file_list(self):
        if not self.file_frame:
            return
        
        self.file_widgets.clear()

        for file in self.selected_files:
            item_frame = ctk.CTkFrame(self.file_frame)
            self.file_widgets[file] = item_frame
            item_frame.pack(fill="x", padx=10, pady=5)
            
            # Diviser le frame en 3 colonnes : delete, nom, like
            item_frame.grid_columnconfigure(0, weight=0) 
            item_frame.grid_columnconfigure(1, weight=1)
            item_frame.grid_columnconfigure(2, weight=0)

            nom = Path(file).name

            # label.pack(side="left", padx=10)

            delete_btn = ctk.CTkButton(
                item_frame,
                text="",
                image=remove_icon,
                width=30,
                height=30,
                corner_radius=15,
                fg_color="transparent",
                hover=False,
                # hover_color="#ee887f",
                command=lambda f=file: self.remove_file(f)
            )
            delete_btn.grid(row=0, column=0, padx=10)

            label = ctk.CTkLabel(item_frame, text=nom, anchor="center")
            label.grid(row=0, column=1)

            like_state = self.selected_files[file]
            like_btn = ctk.CTkButton(
                item_frame,
                text="",
                width=30,
                height=30,
                image=heart_on if like_state else heart_off,
                corner_radius=15,
                fg_color="transparent",
                hover=False,
                # hover_color="#F36C19",
                # command=lambda f=file, b=like_btn: self.toggle_like(f, b)
            )
            like_btn.configure(command=lambda f=file, b=like_btn: self.toggle_like(f, b))
            like_btn.grid(row=0, column=2, padx=10)

    def refresh_file_list(self):
        if not self.file_frame:
            return

        search_text = self.search_entry.get().lower()

        for file, frame in self.file_widgets.items():
            nom = Path(file).name.lower()

            if search_text and search_text not in nom:
                frame.pack_forget()  
            else:
                frame.pack(fill="x", padx=10, pady=5) 

    def update_start_button(self):
        file_count = len(self.selected_files)

        # compter les likes
        likes_count = sum(self.selected_files.values())

        # aucun fichier -> cacher bouton
        if file_count == 0:
            self.bottom_bar.grid_forget()
            return

        # Afficher le bottom bar si pas déjà fait
        if not self.bottom_bar.winfo_ismapped():
            self.bottom_bar.grid(row=3, column=0, sticky="ew", pady=20)

        # afficher bouton si fichiers présents
        if not self.start_button.winfo_ismapped():
            self.start_button.pack(side="left", padx=10)

        # activer/désactiver selon likes
        self.start_button.configure(state="normal" if likes_count >= 3 else "disabled")


    def add_new_files_to_ui(self, files):
        if not self.file_frame:
            return

        for file in files:

            # éviter doublons UI
            if file in self.file_widgets:
                continue

            item_frame = ctk.CTkFrame(self.file_frame)
            self.file_widgets[file] = item_frame
            item_frame.pack(fill="x", padx=10, pady=5)

            item_frame.grid_columnconfigure(0, weight=0)
            item_frame.grid_columnconfigure(1, weight=1)
            item_frame.grid_columnconfigure(2, weight=0)

            nom = Path(file).name

            delete_btn = ctk.CTkButton(
                item_frame,
                text="",
                image=remove_icon,
                width=30,
                height=30,
                corner_radius=15,
                fg_color="transparent",
                hover=False,
                command=lambda f=file: self.remove_file(f)
            )
            delete_btn.grid(row=0, column=0, padx=10)

            label = ctk.CTkLabel(item_frame, text=nom, anchor="center")
            label.grid(row=0, column=1)

            like_state = self.selected_files[file]

            like_btn = ctk.CTkButton(
                item_frame,
                text="",
                width=30,
                height=30,
                image=heart_on if like_state else heart_off,
                corner_radius=15,
                fg_color="transparent",
                hover=False
            )

            like_btn.configure(
                command=lambda f=file, b=like_btn: self.toggle_like(f, b)
            )

            like_btn.grid(row=0, column=2, padx=10)

    def show_search_bar(self):
        self.top_bar.grid(row=0, column=0, sticky="ew", pady=10)
        self.search_visible = True

    def hide_search_bar(self):
        self.top_bar.grid_forget()
        self.search_visible = False

    def start(self):
        # Désactiver le bouton et afficher le loader
        self.start_button.configure(state="disabled")
        self.show_loader()
        
        thread = threading.Thread(target=self._run_recommendation, daemon=True)
        thread.start()

    def _run_recommendation(self):
        """Cette fonction s'exécute dans un thread séparé pour éviter de bloquer l'interface utilisateur.
        Elle gère tout le processus de recommandation, de l'extraction des embeddings à la génération de la playlist, et met à jour l'interface en conséquence."""
        try:

            # Initialiser la base de données, charger le modèle et générer la playlist
            table = initialize_database(db_path="./MusicRecommenderDB")
            session = load_cnn14(onnx_path="cnn14_int8.onnx")
            playlist = recommend_playlist(
                path_dict=self.get_selected_files(),
                session=session,
                table=table,
                lambda_mmr=0.7
            )
            make_m3u(playlist_paths=playlist, output_path="playlist.m3u8")

            # Mettre à jour l'interface dans le thread principal
            self.after(0, self._on_recommendation_success)

        # Gérer les exceptions pour éviter que le thread ne plante silencieusement et pour informer l'utilisateur en cas d'erreur
        except Exception as e:

            self.after(0, partial(self._on_recommendation_error,e))

    def _on_recommendation_success(self):
        """"Cette fonction est appelée dans le thread principal une fois que la recommandation est terminée avec succès. Elle met à jour l'interface utilisateur pour informer l'utilisateur et tente de lancer VLC."""
        
        # Cacher le loader et afficher un message de succès
        self.hide_loader()
        show_toast(self, "Playlist générée : playlist.m3u8")

        # Tenter de lancer VLC pour lire la playlist
        try:
            subprocess.Popen(["vlc", "playlist.m3u8"])
            show_toast(self, "VLC lancé avec succès !")
        except FileNotFoundError:
            vlc_windows_path = Path("C:/Program Files/VideoLAN/VLC/vlc.exe")
            if vlc_windows_path.exists():
                subprocess.Popen([str(vlc_windows_path), "playlist.m3u8"])
                show_toast(self, "VLC lancé avec succès !")
            else:
                show_toast(self, "Erreur : VLC introuvable dans le PATH.")
        except Exception as e:
            show_toast(self, f"Erreur imprévue : {e}")

        # Quitter l'application après un délai pour laisser le temps à l'utilisateur de lire le toast
        self.after(6000, self.destroy)

    def _on_recommendation_error(self, error):
        """Cette fonction est appelée dans le thread principal si une erreur survient pendant le processus de recommandation. Elle met à jour l'interface pour informer l'utilisateur de l'erreur."""

        # Cacher le loader et réactiver le bouton start
        self.hide_loader()
        self.start_button.configure(state="normal")

        # Afficher un message d'erreur à l'utilisateur
        show_toast(self, f"Erreur : {error}")