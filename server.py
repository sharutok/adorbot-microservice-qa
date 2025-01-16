import time
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
from dotenv import load_dotenv
from embeddings import get_embedding_function
import tiktoken

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


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
        # if utility.has_collection(collection_name):
        #     utility.drop_collection(collection_name)

        # Create new collection
        fields = [
            FieldSchema(name="id",dtype=DataType.VARCHAR,max_length=36,is_primary=True,auto_id=False),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536, 
            ),
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

        token_count = get_token_count(
            question, model="text-embedding-3-small"
        )

        print(token_count)

        embedding = embeddings.embed_query(question)

        collection.insert(
            [{"id": id, "question": question, "answer": answer, "embedding": embedding}]
        )
        collection.flush()  # Ensure the data is persisted

        return (jsonify({"message": "Q&A pair added successfully","id": id,"status": "success",}),201)

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to add Q&A pair"}), 500

'''
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
            limit=7,
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
'''


@app.route("/query", methods=["POST"])
def query_database():
    """Query multiple matches using IP metric type with similarity scores."""
    try:
        t1 = time.time()
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Missing query field"}), 400

        # Get limit parameter from request, default to 7 if not specified
        limit = data.get("limit", 7)

        query = data["query"]
        embeddings = get_embedding_function()
        embedding = embeddings.embed_query(query)

        # Search parameters using IP metric type
        search_params = {
            "metric_type": "IP",  # Matching the index metric type
            "params": {
                "nprobe": 10,  # Number of clusters to search
            },
        }

        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["question", "answer"],
        )
        t2 = time.time()
        print("{} secs".format((t2 - t1)))
        if not results or not results[0]:
            return (
                jsonify(
                    {
                        "message": "No matching results found",
                        "status": "success",
                        "data": None,
                    }),404,)

        # Process all matches
        matches = []
        for hit in results[0]:
            similarity_score= float(hit.score) 
            if similarity_score >=float(0.5):
                matches.append(
                    {
                        "pdf_name": hit.entity.get("question"),
                        "pdf_link": hit.entity.get("answer"),
                        # "similarity_score": float(hit.score),  
                    }
                )
        data=formated_response(matches)
        return (
            jsonify({"data": data}),
            200,
        )
        # return (
        #     jsonify(
        #         {
        #             "status": "success",
        #             "data": {
        #                 "matches": matches,
        #                 "total_matches": len(matches),
        #                 "query": query,
        #             },
        #         }
        #     ),
        #     200,
        # )

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to process query"}), 500

def formated_response(response):
    PROMPT_TEMPLATE = """
                        You are an assistant tasked with formatting data into a clear and organized number-point list. Each item should include the name of the PDF and its corresponding link only.
                        For Example:
                        Lord Of the Rings : www.somelink.com
                        Here is the data:
                        {data}
                        Format the data as follows:
                        - pdf_name: pdf_link
                        Ensure clarity and maintain the specified format as given in the example.
                        """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(data=response)
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    response_text = model.invoke(prompt)
    return response_text.content


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
        # utility.list_collections()
        return (
            jsonify(
                {
                    "status": "healthy",
                    "message": "Service is running",
                    # "collection_name": collection_name,
                    # "collection_loaded": collection.is_loaded,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


def get_token_count(text: str, model: str = "text-embedding-3-small") -> int:
    try:  
        encoding = tiktoken.encoding_for_model(model)
        # Encode the text to count the tokens
        tokens = encoding.encode(text)
        print(tokens)
        return len(tokens)

    except Exception as e:
        raise RuntimeError(f"Failed to calculate token count: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
