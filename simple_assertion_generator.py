import openai
import json
import pandas as pd
import logging
from datetime import datetime
from typing import Tuple, Dict
import os
from dotenv import load_dotenv

# Setup
import dotenv
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()
###
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_assertions(sv_code: str, context: str, target_count: int = 10) -> Tuple[Dict, pd.DataFrame]:
    """
    Generate and validate SystemVerilog assertions in a single step.
    """
    prompt = f"""
    You are a hardware verification expert. Generate exactly {target_count} unique SystemVerilog assertions
    for the provided SystemVerilog code. IMPORTANT:
    
    Design Context:
    {context}
    
    Requirements:
    1. Generate a mix of PASS and FAIL assertions (approximately 50/50 split), do not make the assertions too complex.
    2. Each assertion MUST include ALL of these fields:
       - "name": unique identifier for the assertion
       - "assertion_sv": ONLY the assertion code, no comments or explanations
       - "line_number": the line number directly linked to the assertion. ONLY use line numbers that exist in the provided code file
       - "expected_result": either "PASS" or "FAIL"
    3. Use only signals and states that exist in the provided code
    4. DO NOT add any comments to the assertions or to the line numbers
    5. DO NOT generate or assume any code that isn't in the original file
    6. DO NOT use temporal operators like ##1 or ##
    7. DO NOT add safety bug comments or any other annotations to the line numbers
    8. Count the actual lines in the provided code file for line numbers
    
    Respond ONLY with valid JSON. Example format:
    {{
        "assertions": [
            {{
                "name": "check_idle_to_ready",
                "assertion_sv": "assert property (@(posedge clk) state == IDLE |-> state == READY)",
                "line_number": 42,
                "expected_result": "PASS"
            }}
        ]
    }}

    IMPORTANT:
    - Assertions should be clean, without comments
    - Keep assertions simple without temporal operators
    - Use only immediate implications (|->)
    - No ##1 or other cycle delays
    - Line numbers must correspond to actual lines in the provided code, and the line must be non-empty.
    - Do not make assumptions about code outside what is provided
    - Ensure generated data will be compatible with a CSV file.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"SystemVerilog Code:\n{sv_code}"}
            ],
            temperature=0.7
        )

        logger.info(f"Received response from OpenAI API")
        logger.info(f"SystemVerilog code length: {len(sv_code)}")
        
        content = response.choices[0].message.content
        
        # Print raw response for debugging
        print("\nRaw API Response:")
        print(content)
        
        try:
            content = content.strip()
            result = json.loads(content)
            
            # Handle both single assertion and array of assertions
            if "assertions" in result:
                assertions = result["assertions"]
            elif "name" in result:
                assertions = [result]
            else:
                assertions = []
            
        except json.JSONDecodeError as je:
            logger.error(f"JSON parsing error: {str(je)}")
            logger.error(f"Problematic content: {content}")
            return {"assertions": [], "count": 0}, pd.DataFrame(columns=["name", "assertion_sv", "line_number", "expected_result"])

        # Validate assertion format
        required_fields = ["name", "assertion_sv", "line_number", "expected_result"]
        valid_assertions = []
        
        for assertion in assertions:
            if all(field in assertion for field in required_fields):
                valid_assertions.append(assertion)
            else:
                missing_fields = [field for field in required_fields if field not in assertion]
                logger.warning(f"Assertion '{assertion.get('name', 'unnamed')}' missing fields: {missing_fields}")

        # Create DataFrame
        if valid_assertions:
            df = pd.DataFrame(valid_assertions)
            
            # Save to CSV with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #df.to_csv(f"generated_assertions_{timestamp}.csv", index=False)
        else:
            df = pd.DataFrame(columns=required_fields)
            logger.warning("No valid assertions generated")

        results = {
            "assertions": valid_assertions,
            "count": len(valid_assertions)
        }

        return results, df

    except Exception as e:
        logger.error(f"Error generating assertions: {str(e)}")
        return {"assertions": [], "count": 0}, pd.DataFrame(columns=["name", "assertion_sv", "line_number", "expected_result"])

if __name__ == "__main__":
    # Read SystemVerilog file
    file_path = '/Users/sanjayravindran/Documents/566Project/ORIGINAL_CODE/test_2/test_2.txt'
    try:
        with open(file_path, 'r') as file:
            sv_code = file.read()
            logger.info(f"Successfully read file. Content length: {len(sv_code)}")
            if len(sv_code) == 0:
                raise ValueError("File is empty")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        exit(1)
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        exit(1)

    # Define context for the design
    context = """
    This is a SystemVerilog design file.
    Key aspects to verify:
    1. State machine transitions and properties
    2. Signal timing relationships
    3. Protocol compliance
    4. Interface requirements
    5. Edge cases and corner conditions
    Use Case: This module is ideal for simulating and testing coin-based vending machines. It ensures accurate handling of deposits, change, and refunds while maintaining a clear state-based operational flow.


    """
    # Generate assertions
    results, df = generate_assertions(sv_code, context)

    # Add original code column if we have valid assertions
    if not df.empty:
        # Split the code into lines
        code_lines = sv_code.splitlines()
        
        def mark_line_in_code(code: str, line_num: int) -> str:
            lines = code.splitlines()
            target_line = line_num - 1 # Convert to 0-based index
            
            # Add comment marker to the target line
            if 0 <= target_line < len(lines):
                if '//' in lines[target_line]:
                    parts = lines[target_line].split('//', 1)
                    lines[target_line] = parts[0].rstrip() + " // "
                else:
                    lines[target_line] = lines[target_line].rstrip() + " // "
            
            # Return the full code with the marked line
            return '\n'.join(lines)
        
        # Add the marked code to the DataFrame
        df['code'] = df.apply(lambda row: mark_line_in_code(sv_code, row['line_number']), axis=1)
        
        # Save DataFrame
        df.to_csv("/Users/sanjayravindran/Documents/566Project/DATA/GENERATED/GENERATED/test_2/test2_data.csv", index=False)
    else:
        print("\nNo valid assertions were generated. Please try again.")