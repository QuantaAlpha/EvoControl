import chromadb
import os
from argparse import ArgumentParser

# Define the path to the ChromaDB directory
# The path provided by the user was: \data\CodeEfficiency\SE-Agent\trajectories_perf\deepseek-v3\deepseek-v3-test-global_memory-example_20251206_133305\global_memory
# Converting to Linux path format
parser = ArgumentParser()
parser.add_argument("--db_dir", type=str)
args = parser.parse_args()

PERSIST_DIRECTORY = args.db_dir


def inspect_chromadb(persist_directory):
    print(f"Inspecting ChromaDB at: {persist_directory}")

    if not os.path.exists(persist_directory):
        print(f"Error: Directory does not exist: {persist_directory}")
        return

    try:
        # Initialize the ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)

        # List all collections
        collections = client.list_collections()
        print(f"\nFound {len(collections)} collections:")

        for collection in collections:
            print(f"\n--- Collection: {collection.name} ---")
            print(f"ID: {collection.id}")
            print(f"Metadata: {collection.metadata}")

            # Get basic stats about the collection
            count = collection.count()
            print(f"Item count: {count}")

            if count > 0:
                # Peek at the first few items (limit to 5)
                print("Peeking at first 5 items:")
                peek_data = collection.peek(limit=5)

                # Pretty print the peek data
                ids = peek_data.get("ids", [])
                documents = peek_data.get("documents", [])
                metadatas = peek_data.get("metadatas", [])
                embeddings = peek_data.get("embeddings", [])

                for i in range(len(ids)):
                    print(f"\n  Item {i + 1}:")
                    print(f"    ID: {ids[i]}")
                    if embeddings.any() and i < len(embeddings):
                        print(f"    Embeddings: {embeddings[i]}")
                    if metadatas and i < len(metadatas):
                        print(f"    Metadata: {metadatas[i]}")
                    if documents and i < len(documents):
                        # Truncate document content if it's too long
                        doc_preview = documents[i]
                        print(f"    Document: \n{doc_preview}")
            else:
                print("Collection is empty.")

    except Exception as e:
        print(f"An error occurred while inspecting ChromaDB: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    inspect_chromadb(PERSIST_DIRECTORY)
