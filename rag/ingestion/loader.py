from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents(path):
    loader = DirectoryLoader(
        path,
        glob="**/*.md",
        loader_cls=TextLoader
    )
    docs = loader.load()

    return docs 