import email
import json
import boto3
from typing import Dict, Any
import time

class EmlReader:
    """Reads .eml file and extracts text content."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_eml(self) -> str:
        """Reads .eml file and returns its text content."""
        with open(self.file_path, 'r') as file:
            msg = email.message_from_file(file)
            
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    return part.get_payload(decode=True).decode()
        else:
            return msg.get_payload(decode=True).decode()

        return ""  # Return empty string if no text content found

class BedrockClaudeClient:
    """Handles interaction with AWS Bedrock Claude Sonnet model."""

    def __init__(self, region: str, model_id: str = "arn:aws:bedrock:us-east-1:762454182244:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def invoke_model(self, email_content: str) -> Dict[str, Any]:
        """Send email content to Claude Sonnet and return JSON response."""
        prompt = f"""
        Please extract all relevant information, including the reclass system, account entries from the following email content and return them in a structured nested JSON format.
        Do not include any explanatory text before or after the JSON. The response should be valid, parseable JSON only.

        Email Content:
        {email_content}
        """

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        
        content = result['content'][0]['text']

        # Find the start and end of the JSON in the content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1

        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return {"error": "Unable to parse JSON", "raw_output": content}
        else:
            return {"error": "No JSON found in the response", "raw_output": content}

class BedrockExtractionAgent:
    """Main pipeline to process .eml and get Claude Sonnet response."""

    def __init__(self, eml_file: str, region: str):
        self.eml_file = eml_file
        self.region = region

    def execute(self , ) -> Dict[str, Any]:
        reader = EmlReader(self.eml_file)
        email_content = reader.read_eml()

        bedrock_client = BedrockClaudeClient(self.region)
        response = bedrock_client.invoke_model(email_content)
        return response

if __name__ == "__main__":


    start_time = time.time()
    eml_file_path = r"C:\subesh\works\R2R\Proposed_Design\MJE_Accrual_Asistant\sample_email\_MEU_DEMO  _  Email Test by Suneel-1.eml"  # Replace with your .eml file path
    region = "us-east-1"  # Replace with your AWS region

    agent = BedrockExtractionAgent(eml_file_path, region)
    output = agent.execute()

    if isinstance(output, dict) and "error" in output:
        print("Error:", output["error"])
        print("Raw output:", output["raw_output"])
    else:
        print(json.dumps(output, indent=2))

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.4f} seconds")
