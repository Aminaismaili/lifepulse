import os
import pdfplumber
from langchain_chroma import Chroma  # Mise à jour de l'import Chroma
from langchain_ollama import OllamaEmbeddings  # Mise à jour de l'import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Répertoire des PDF
pdf_directory = "c:/Users/hp/Desktop/DATA2"

# Fonction pour sauvegarder la progression
def save_page_progress(file_path, page_num):
    """Sauvegarde la progression de la page extraite dans un fichier journal."""
    with open("progress_page.log", "w") as log_file:
        log_file.write(f"{file_path},{page_num}\n")

# Fonction pour charger la progression
def load_page_progress():
    """Charge la progression de la dernière page extraite depuis le fichier journal."""
    if os.path.exists("progress_page.log"):
        with open("progress_page.log", "r") as log_file:
            content = log_file.read().strip()
            if content:
                file_path, page_num = content.split(",")
                return file_path, int(page_num)
    return None, 0

# Réinitialiser la progression
def reset_progress():
    """Supprime le fichier de suivi de progression."""
    if os.path.exists("progress_page.log"):
        os.remove("progress_page.log")
    print("Progression réinitialisée.")

# Fonction pour lire un PDF avec gestion de la progression
def read_pdf(file_path):
    """Lit un fichier PDF et extrait son contenu, avec reprise de la progression."""
    last_file, last_page = load_page_progress()  # Charger la progression
    if last_file == file_path:
        start_page = last_page + 1  # Reprendre après la dernière page traitée
    else:
        start_page = 0  # Commencer depuis le début si un nouveau fichier

    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page_num, page in enumerate(pdf.pages):
                if page_num < start_page:  # Ignorer les pages déjà traitées
                    continue

                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"  # Ajout de texte
                    print(f"Page {page_num + 1}/{len(pdf.pages)} extraite avec succès.")
                    
                    # Sauvegarder la progression après chaque page
                    save_page_progress(file_path, page_num)
                except Exception as e:
                    print(f"Erreur lors de l'extraction de la page {page_num + 1} dans {file_path}: {e}")
                    save_page_progress(file_path, page_num)  # Sauvegarder l'erreur pour reprendre
            return text
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier PDF {file_path}: {e}")
        return ""  # Retourne un texte vide si une erreur survient

# Charger les documents PDF depuis un répertoire
def load_documents_from_directory(directory_path):
    """Charge les documents PDF en reprenant la progression de pages si nécessaire."""
    files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".pdf")]
    documents = []
    for file in files:
        print(f"Traitement du fichier : {file}")
        content = read_pdf(file)
        if content.strip():  # Vérifie si le contenu n'est pas vide
            documents.append(Document(page_content=content))
        else:
            print(f"Aucun contenu extrait du fichier {file}.")
    return documents

# Charger tous les documents
def load_all_documents():
    """Charge tous les documents du répertoire."""
    return load_documents_from_directory(pdf_directory)

# Fonction pour ingérer dans une base de vecteurs
def ingest_into_vector_store(combined_texts):
    # Diviser le texte en chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200,
        separator=".",  # Utilisez le point comme séparateur principal
        keep_separator=False  # Respectez strictement la taille limite
    )
    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(
        [Document(page_content=text.replace("\n", ". ")) for text in combined_texts]
    )

    # Vérification de la taille de chaque chunk avant de les ajouter au vecteur store
    for chunk in doc_splits:
        token_count = len(chunk.page_content.split())  # Compter les tokens en fonction des espaces
        print(f"Chunk size (tokens): {token_count}")
        if token_count > 2000:
            print("Warning: Chunk size exceeds 2000 tokens!")

    # Initialize the Chroma vector store with a specific collection name
    db = Chroma(
        persist_directory="c:/Users/hp/Desktop/CHAT2",
        embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        collection_name="rag-chroma"
    )

    # Ajouter les documents à Chroma
    db.add_documents(doc_splits)
    print("Données ingérées dans la base de données vectorielle.")

# Initialiser le vecteur store pour recherche
def initialize_vector_store():
    """Initialise le vecteur store pour récupération."""
    db = Chroma(
        persist_directory="c:/Users/hp/Desktop/chatdb",
        embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        collection_name="rag-chroma"
    )
    return db

# Fonction principale
def main():
    """Fonction principale qui charge et ingère les documents."""
    all_documents = load_all_documents()
    if all_documents:
        combined_texts = [doc.page_content for doc in all_documents]
        ingest_into_vector_store(combined_texts)
    else:
        print("Aucune donnée à traiter.")

# Vérifier si la fonction main() a déjà été appelée avant de l'exécuter
if __name__ == "__main__":
    main()