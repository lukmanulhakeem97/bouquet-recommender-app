import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import os
from datasets import load_dataset
from PIL import Image
import base64

class FlowerRAG:
    def __init__(self, 
                 dataset_folder="./data/images/flowers-102-categories", 
                 db_path="./data/dbs/flower.db"):
        
        self.dataset_folder = dataset_folder
        self.db_path = db_path

        if not os.path.exists(self.dataset_folder):
            self.load_flower_dataset(self.dataset_folder)
        
        self.flower_collection = self.load_rag_db(self.db_path)
        self.populate_rag_db(self.dataset_folder)

    def load_rag_db(self, db_path):
        # Define ChromaDB for image search
        chroma_client = chromadb.PersistentClient(path=db_path)
        image_loader = ImageLoader()
        embedding_function = OpenCLIPEmbeddingFunction()
        collection = chroma_client.get_or_create_collection(
            "flowers_collection",
            embedding_function=embedding_function,
            data_loader=image_loader,
        )

        return collection
    
    def populate_rag_db(self, dataset_folder):
        ids = []
        uris = []
        # Iterate over each file in the dataset folder -- uncomment this for the first time run ###
        for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
            if filename.endswith(".png"):
                file_path = os.path.join(dataset_folder, filename)
                image = Image.open(file_path)

                # Append id and uri to respective lists
                ids.append(str(i))
                uris.append(file_path)

        # Assuming multimodal_db is already defined and available
        self.flower_collection.add(ids=ids, uris=uris)

    # Query the VectorDB (ChromaDB) for images based on text query
    def query_db(self, query, results=2):
        res = self.flower_collection.query(
            query_texts=[query], n_results=results, include=["uris", "distances"]
        )
        return res

    # Function to save images
    def save_images(self, dataset, dataset_folder, num_images=500):
        for i in range(num_images):
            print(f"Saving image {i+1} of {num_images}")
            # Get the image data
            image = dataset["train"][i]["image"]

            # Save the image
            image.save(os.path.join(dataset_folder, f"flower_{i+1}.png"))

    def load_flower_dataset(self, dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)
        ds = load_dataset("huggan/flowers-102-categories")
        self.save_images(ds, dataset_folder, num_images=500)
        print("Dataset loaded and images saved.")
        


# Helper function to format inputs for the prompt
def format_prompt_inputs(data, user_query):
    inputs = {}
    inputs["user_query"] = user_query

    # Get first two image paths
    image_path_1 = data["uris"][0][0]
    image_path_2 = data["uris"][0][1]

    # Encode images to base64
    with open(image_path_1, "rb") as image_file:
        image_data_1 = image_file.read()
    inputs["image_data_1"] = base64.b64encode(image_data_1).decode("utf-8")

    with open(image_path_2, "rb") as image_file:
        image_data_2 = image_file.read()
    inputs["image_data_2"] = base64.b64encode(image_data_2).decode("utf-8")

    return inputs