import os 

path="app/data/pdfs/Animal.pdf"
if os.path.exists(path):
    print("File exists")
else:
    print("File does not exist")