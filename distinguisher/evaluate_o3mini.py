import os
from openai import OpenAI
import pandas as pd
import logging
from distinguisher.models.base import Distinguisher

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Create an OpenAI client instance using the new API interface
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class OpenAIDistinguisher(Distinguisher):
    @staticmethod
    def prompt_fn(persona, **kwargs):
        messages = []
        if persona:
            messages.append({"role": "system", "content": persona})
            
        messages.append({
            "role": "user",
            "content": f"""
            Modified text: {kwargs["P"]}
            ___________________________________________________________
            Original response A: {kwargs["A"]}
            __________________________________________________________
            Original response B: {kwargs["B"]}
            __________________________________________________________
            Which response does the modified text come from? Reply with only one character.
            """
        })
        
        return messages

    def apply_prompt(self, datarow, flip, prefix=""):
        # Gather input keys from the row and add the origins.
        inputs = {k: datarow[k] for k in self.input_keys}
        inputs['A'] = self.origin_A
        inputs['B'] = self.origin_B
        
        if flip:
            inputs['A'], inputs['B'] = inputs['B'], inputs['A']
            prefix += "flipped_"

        try:
            messages = self.prompt_fn(persona=self.persona, **inputs)
            response = client.chat.completions.create(
                model="o3-mini",
                messages=messages,
            )
            print(response)
            choice = response.choices[0].message.content.strip().upper()
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            choice = "FAIL"

        output_dict = {prefix + "choice": choice if choice in ["A", "B"] else "FAIL"}
        
        if flip:
            self.flip_choices(output_dict, prefix)
            
        return output_dict

    @property
    def input_keys(self):
        return ["P"]

    @property
    def output_keys(self):
        return ["choice"]

def evaluate3(distinguisher, rows, output_path, skip=0):
    """Modified evaluation function for OpenAI API."""
    for i, row in rows.iterrows():
        if i < skip:
            continue
        logging.info(f"Processing row {i+1}/{len(rows)}")
        distinguisher.set_origin(row['origin_A'], row['origin_B'])
        
        try:
            output = distinguisher.distinguish_row(row, prefix="3rd_")
            for key, value in output.items():
                rows.at[i, key] = value
        except Exception as e:
            logging.error(f"Failed on row {i}: {str(e)}")
            continue
            
        pd.DataFrame([rows.iloc[i]]).to_csv(
            output_path, 
            mode='a', 
            header=not os.path.exists(output_path), 
            index=False
        )

def main():
    # Define a persona for the distinguisher.
    persona = """
    You are an expert in analyzing the similarity of responses.
    You specialize in detecting whether one response has been derived from another by carefully analyzing the content and structure of the text.
    Your expertise in linguistic analysis allows you to distinguish which responses are the most closely related.
    Your goal is to provide a clear, concise, and accurate assessment of the provided instructions.
    # """
    # Note: Pass None for the first parameter if the base class expects an LLM instance.
    distinguisher = OpenAIDistinguisher(None, persona)
    
    # Load the dataset (adjust the path as necessary)
    df = pd.read_csv("distinguisher/data/failed_distinguishes_for_llama3.1-70B-long.csv")
    
    # Process the CSV rows and save the results.
    evaluate3(distinguisher, df, "distinguisher/results/redemption_long_with_o3-mini.csv")

if __name__ == "__main__":

    # python -m distinguisher.evaluate_o3mini

    main()
