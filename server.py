from flask import Flask, request, jsonify
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)
import uuid
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Establish Milvus connection with error handling
try:
    connections.connect(host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
    raise

# Environment variable validation
required_env_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

collection_name = "qa_collection"


def initialize_collection():
    """Initialize Milvus collection with proper index settings."""
    try:
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        # Create new collection
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=36,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
        ]

        schema = CollectionSchema(fields, description="Collection for QA data")
        collection = Collection(name=collection_name, schema=schema)

        # Create index with IP metric type
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",  # Using IP (Inner Product) for better semantic search
            "params": {"nlist": 128},
        }

        collection.create_index(field_name="embedding", index_params=index_params)

        print("Collection initialized successfully with IP metric type index")
        return collection
    except Exception as e:
        print(f"Failed to initialize collection: {e}")
        raise


collection = initialize_collection()
collection.load()


def get_embedding_function():
    try:
        return OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {e}")


@app.route("/add", methods=["POST"])
def add_qa_pair():
    """Add a Q&A pair with proper error handling."""
    try:
        data = request.json
        if not data or "question" not in data or "answer" not in data:
            return jsonify({"error": "Missing required fields"}), 400

        question = data["question"]
        answer = data["answer"]
        id = str(uuid.uuid4())

        embeddings = get_embedding_function()
        embedding = embeddings.embed_query(question)

        insert_result = collection.insert(
            [{"id": id, "question": question, "answer": answer, "embedding": embedding}]
        )
        collection.flush()  # Ensure the data is persisted

        return (jsonify({"message": "Q&A pair added successfully","id": id,"status": "success",}),201)

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to add Q&A pair"}), 500


@app.route("/query", methods=["POST"])
def query_database():
    """Query the closest match using IP metric type."""
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Missing query field"}), 400

        query = data["query"]
        embeddings = get_embedding_function()
        embedding = embeddings.embed_query(query)

        # Search parameters using IP metric type
        search_params = {
            "metric_type": "IP",  # Matching the index metric type
            "params": {"nprobe": 10},
        }

        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=1,
            output_fields=["question", "answer"],
        )

        if not results or not results[0]:
            return (
                jsonify(
                    {
                        "message": "No matching results found",
                        "status": "success",
                        "data": None,
                    }
                ),
                404,
            )

        match = results[0][0]
        return (
            jsonify(
                {
                    "status": "success",
                    "data": {
                        "question": match.entity.get("question"),
                        "answer": match.entity.get("answer"),
                        "similarity_score": float(
                            match.score
                        ),  # Convert to float for JSON serialization
                    },
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to process query"}), 500


@app.route("/update", methods=["PUT"])
def update_qa_pair():
    """Update a Q&A pair with error handling."""
    try:
        data = request.json
        if not data or not all(k in data for k in ["id", "question", "answer"]):
            return jsonify({"error": "Missing required fields"}), 400

        id = data["id"]
        new_question = data["question"]
        new_answer = data["answer"]

        # Check if the record exists
        expr = f"id == '{id}'"
        if collection.query(expr=expr, output_fields=["id"]) == []:
            return jsonify({"error": "Q&A pair not found"}), 404

        # Delete existing record
        collection.delete(expr)

        # Create new embedding
        embeddings = get_embedding_function()
        embedding = embeddings.embed_query(new_question)

        # Insert updated record
        collection.insert(
            [
                {
                    "id": id,
                    "question": new_question,
                    "answer": new_answer,
                    "embedding": embedding,
                }
            ]
        )
        collection.flush()  # Ensure the update is persisted

        return (
            jsonify({"message": "Q&A pair updated successfully", "status": "success"}),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to update Q&A pair"}), 500


@app.route("/delete", methods=["DELETE"])
def delete_qa_pair():
    """Delete a Q&A pair with error handling."""
    try:
        data = request.json
        if not data or "id" not in data:
            return jsonify({"error": "Missing ID field"}), 400

        id = data["id"]
        expr = f"id == '{id}'"

        # Check if the record exists
        if collection.query(expr=expr, output_fields=["id"]) == []:
            return jsonify({"error": "Q&A pair not found"}), 404

        collection.delete(expr)
        collection.flush()  # Ensure the deletion is persisted

        return (
            jsonify({"message": "Q&A pair deleted successfully", "status": "success"}),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to delete Q&A pair"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint with connection testing."""
    try:
        # Test Milvus connection
        utility.list_collections()
        return (
            jsonify(
                {
                    "status": "healthy",
                    "message": "Service is running",
                    "collection_name": collection_name,
                    "collection_loaded": collection.is_loaded,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
