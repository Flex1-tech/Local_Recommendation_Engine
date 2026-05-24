import interface 


app = interface.App()
app.after(200, lambda: app.iconbitmap('assets/VLC.ico')) 
app.mainloop()
