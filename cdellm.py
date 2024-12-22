import streamlit as st
import os 
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from  data22  import initialize_vector_store


# function to read the pdf file
def read_pdf(file):
    pdfReader = PyPDF2.PdfReader(file)
    all_page_text = ""
    for page in pdfReader.pages:
        all_page_text += page.extract_text() + "\n"
    return all_page_text


def retrieve_from_db(question):
    # get the model
    model = ChatOllama(model="mistral")
    # initialize the vector store
    db = initialize_vector_store()

    retriever = db.similarity_search(question, k=2)
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    "
    """

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    after_rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | after_rag_prompt
        | model
        | StrOutputParser()
    )

    return after_rag_chain.invoke({"context": retriever, "question": question})


def retriever(doc, question):
    model_local = ChatOllama(model="mistral")
    doc = Document(page_content=doc)
    doc = [doc]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(doc)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    )
    retriever = vectorstore.as_retriever(k=2)
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(question)
 

# --- Définir les messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Fonctionnalité d'appel urgent ---
if "phone_number" not in st.session_state:
    st.session_state.phone_number = None  # Initialiser le numéro de téléphone

# Vérifier si la clé 'user_info' existe, sinon l'initialiser
if "user_info" not in st.session_state:
    st.session_state.user_info = None  # Informations de l'utilisateur



# --- Titre principal ---
st.title("Pulse Life 🫀")
st.write("This is a RAG chatbot that can answer questions based on a given context.")

# --- Barre latérale pour nouvelle discussion ---
with st.sidebar:
    if st.button("➕", help="Nouvelle Discussion"):
        # Lorsque le bouton est cliqué, réinitialiser la session des messages
        st.session_state.messages = []
        st.write("Nouvelle discussion commencée.")
    
    # Bouton "Appel urgent" avec icône de téléphone
    if st.button("📞 Appel urgent", help="Cliquez pour appeler un numéro urgent"):
        # Si un numéro est déjà enregistré
        if st.session_state.phone_number:
            # Afficher le numéro enregistré
            st.write(f"Numéro enregistré : {st.session_state.phone_number}")
            st.write("Cliquez ci-dessous pour appeler ce numéro.")
            
            # Bouton pour simuler l'appel
            if st.button("Appeler le numéro enregistré"):
                st.write(f"Appel en cours vers le numéro : {st.session_state.phone_number}")
            
        else:
            # Si aucun numéro n'est enregistré, demander à l'utilisateur d'entrer un numéro
            st.write("Entrez votre numéro de téléphone pour l'appel urgent :")
            phone_number_input = st.text_input("Numéro de téléphone", value="")
            
            # Bouton pour enregistrer le numéro
            if st.button("Enregistrer le numéro"):
                if phone_number_input:
                    st.session_state.phone_number = phone_number_input
                    st.write(f"Numéro enregistré : {st.session_state.phone_number}")
                else:
                    st.warning("Veuillez entrer un numéro de téléphone valide.")
    
    # Bouton pour s'inscrire ou se connecter
    if not st.session_state.user_info:
        st.title("Se connecter ou s'inscrire")
        option = st.radio("Choisir une option", ["S'inscrire", "Se connecter avec Google"])

        if option == "S'inscrire":
            # Formulaire d'inscription
            st.subheader("Formulaire d'inscription")
            username = st.text_input("Nom d'utilisateur")
            email = st.text_input("Adresse e-mail")
            password = st.text_input("Mot de passe", type="password")
            
            if st.button("S'inscrire"):
                if username and email and password:
                    # Enregistrer l'utilisateur
                    st.session_state.user_info = {"username": username, "email": email, "password": password}
                    st.success("Inscription réussie !")
                else:
                    st.error("Veuillez remplir tous les champs.")
        
        if option == "Se connecter avec Google":
            # Simuler la connexion avec Google
            st.subheader("Connexion Google")
            st.write("Cliquez sur le bouton ci-dessous pour vous connecter avec votre compte Google.")
            google_button = st.button("Se connecter avec Google")
            
            if google_button:
                # Rediriger vers Google OAuth ou procéder à la connexion
                # (Remarque: Streamlit ne supporte pas nativement OAuth pour le moment)
                st.write("Redirection vers Google...")
                # Ici, tu devrais intégrer un vrai processus OAuth ou rediriger vers une page externe.

    else:
        # Si l'utilisateur est connecté
        st.write(f"Bienvenue, {st.session_state.user_info['username']}!") 

# --- Affichage des messages précédents ---
for message in st.session_state.messages:
    st.write(message)

# --- Gestion du fichier PDF ---
file = st.file_uploader("Upload a PDF file", type=["pdf"])
if file:
    doc = read_pdf(file)  # Fonction de lecture du PDF
    question = st.text_input("Ask a question")
    if question:
        if st.button("Ask"):  # Lorsque l'utilisateur appuie sur "Demander"
            answer = retriever(doc, question)  # Fonction de récupération de réponse
            st.write(answer)
            # Ajouter la question et la réponse dans l'historique des messages
            st.session_state.messages.append(f"You: {question}")
            st.session_state.messages.append(f"🩺: {answer}")
else:
    question = st.text_input("Ask a question")
    if question:
        if st.button("Ask"):  # Lorsque l'utilisateur appuie sur "Demander"
            answer = retrieve_from_db(question)  # Fonction de récupération de réponse depuis la base de données
            st.write(answer)
            # Ajouter la question et la réponse dans l'historique des messages
            st.session_state.messages.append(f"You: {question}")
            st.session_state.messages.append(f"🩺: {answer}")