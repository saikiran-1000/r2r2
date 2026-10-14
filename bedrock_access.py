import boto3
import pandas as pd
import json
from typing import Dict, Any


class ExcelReader:
    """Reads Excel file and extracts all sheet data."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_excel(self) -> Dict[str, Any]:
        """Reads all sheets and returns data with sheet names."""
        all_sheets = pd.read_excel(self.file_path, sheet_name=None)

        extracted_data = {}
        for sheet_name, df in all_sheets.items():
            extracted_data[sheet_name] = df.to_dict(orient="records")

        return extracted_data


class BedrockClaudeClient:
    """Handles interaction with AWS Bedrock Claude Sonnet model."""

    def __init__(self, region: str, model_id: str = "arn:aws:bedrock:us-east-1:762454182244:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def invoke_model(self, message: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send dataset to Claude Sonnet and return JSON response."""
        prompt = f"""
        You are given a dataset extracted from multiple Excel sheets.
        Each sheet has been converted into JSON.
        
        Dataset: {json.dumps(data, indent=2)}
        In the each nested record excel sheet name should be included.
        Return Extracted journal entry in a structured nested JSON format. Do not include any explanatory text before or after the JSON. The response should be valid, parseable JSON only.
        """

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            }),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
     
        output_text = result["content"][0]["text"]

        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            return {"raw_output": output_text}


class ExcelToBedrockPipeline:
    """Main pipeline to process Excel and get Claude Sonnet response."""

    def __init__(self, excel_file: str, region: str):
        self.excel_file = excel_file
        self.region = region

    def run(self, message: str) -> Dict[str, Any]:
        reader = ExcelReader(self.excel_file)
        extracted_data = reader.read_excel()

        bedrock_client = BedrockClaudeClient(self.region)
        response = bedrock_client.invoke_model(message, extracted_data)
        return response


if __name__ == "__main__":
    excel_file_path = r"C:\subesh\works\R2R\Proposed_Design\MJE_Accrual_Asistant\sample_email\extract\Legal Services MEU Accrual Submission - Czech Republic.xlsx"   # Replace with your file path
    region = "us-east-1"             # Replace with your AWS region
    message = "Please return this data as clean nested JSON."

    pipeline = ExcelToBedrockPipeline(excel_file_path, region)
    output = pipeline.run(message)

    print(json.dumps(output, indent=2))
