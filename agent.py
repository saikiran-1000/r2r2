import email
import json
import boto3
import pandas as pd
import os
from typing import Dict, Any, List


class EmlReader:
    """Reads .eml file, extracts text content and attachments."""

    def __init__(self, file_path: str, attachments_dir: str = "attachments"):
        self.file_path = file_path
        self.attachments_dir = attachments_dir
        os.makedirs(attachments_dir, exist_ok=True)

    def read_eml(self) -> str:
        """Reads .eml file and returns its text content."""
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as file:
            msg = email.message_from_file(file)

        body_text = ""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition"))
                content_type = part.get_content_type()

                # Extract plain text body
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    body_text += part.get_payload(decode=True).decode(errors="ignore")

                # Extract Excel attachments
                elif "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        filepath = os.path.join(self.attachments_dir, filename)
                        with open(filepath, "wb") as f:
                            f.write(part.get_payload(decode=True))
                        attachments.append(filepath)
        else:
            body_text = msg.get_payload(decode=True).decode(errors="ignore")

        return body_text.strip(), attachments


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

    def __init__(
        self,
        region: str,
        model_id: str = "arn:aws:bedrock:us-east-1:762454182244:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    ):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def invoke_model(self, email_content: str, attachments_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send email + attachment dataset to Claude Sonnet and return JSON response."""
        prompt = f"""
        You are given an email and its attachments.

        Email Content:
        {email_content}

        Attachment Data (from Excel files):
        {json.dumps(attachments_data, indent=2)}

        Task:
        Extract relevant journal entries, account details, and other structured financial info.
        Return as a valid, parseable JSON only (no explanation text).
        """

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                }
            ),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        output_text = result["content"][0]["text"]

        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            return {"raw_output": output_text}


class EmailExtractionAgent:
    """Main pipeline to process .eml, extract body + attachments, and query Claude Sonnet."""

    def __init__(self, eml_file: str, region: str):
        self.eml_file = eml_file
        self.region = region

    def execute(self) -> Dict[str, Any]:
        # Step 1: Read email + attachments
        reader = EmlReader(self.eml_file)
        email_content, attachments = reader.read_eml()

        # Step 2: Process Excel attachments into JSON
        attachments_data = {}
        for attachment in attachments:
            if attachment.endswith((".xlsx", ".xls")):
                excel_reader = ExcelReader(attachment)
                attachments_data[os.path.basename(attachment)] = excel_reader.read_excel()

        # Step 3: Call Bedrock Claude
        bedrock_client = BedrockClaudeClient(self.region)
        response = bedrock_client.invoke_model(email_content, attachments_data)

        return response


if __name__ == "__main__":
    eml_file_path = r"C:\Users\703398864\OneDrive - Genpact\Desktop\python_skydive\R2R\MEU_DEMO _1.eml"
    region = "us-east-1"  # AWS region

    agent = EmailExtractionAgent(eml_file_path, region)
    output = agent.execute()

    print(json.dumps(output, indent=2))
