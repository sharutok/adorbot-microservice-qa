from openai import OpenAI
import time
import os


class OpenAIBatchProcessor:
    def __init__(self, api_key):
        client = OpenAI(api_key=api_key)
        self.client = client

    def process_batch(self, input_file_path, endpoint, completion_window):
        with open(input_file_path, "rb") as file:
            uploaded_file = self.client.files.create(
                file=file,
                purpose="batch"
            )

        batch_job = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint=endpoint,
            completion_window=completion_window
        )


        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)  
            print(f"Batch job status: {batch_job.status}...trying again in 3 seconds...")
            batch_job = self.client.batches.retrieve(batch_job.id)

        if batch_job.status == "completed":
            result_file_id = batch_job.output_file_id
            result = self.client.files.retrieve(result_file_id)
            
            content = self.client.files.content(result.id)
            content_bytes = content.read()
            
            with open("Step2.jsonl", 'wb') as f:
                f.write(content_bytes)
            print(f"File successfully downloaded as {result.filename}")
            return result
        else:
            print(f"Batch job failed with status: {batch_job.status}")
            return None

api_key = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
processor = OpenAIBatchProcessor(api_key)


# Print the UUID
input_file_path = "Step1.jsonl"

endpoint = "/v1/embeddings"
completion_window = "24h"

processor.process_batch(input_file_path, endpoint, completion_window)