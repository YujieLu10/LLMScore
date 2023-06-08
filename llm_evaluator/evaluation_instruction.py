import openai

class EvaluationInstructor:
    def __init__(self, api_key, llm_id="gpt-4"):
        openai.api_key = api_key
        self.llm_id = llm_id
        self.evaluation_instruction_1 = """\nBased on the text prompt and image description, provide the following 2 scores and 2 rationales to explain the scores (Note that "a" in text prompt means numerical one):
        \nX1: Rate the overall quality (scale: 1-100) of the image description in terms of matching the text prompt.
        \nX2: Provide the number of composition errors Y (scale: 0-9) in the image description compared to the text prompt. One error should be counted for each composition error, including incorrect number of each object, incorrect color, location, shape, size, material of each object, and incorrect relationship among objects. If an object category mentioned in the text prompt is missing in the description, count it as 4 errors.
        \nZ1: Explain the overall rating X within one paragraph less than 5 sentences without separator.
        \nZ2: Explain the composition error Y within one paragraph less than 5 sentences without separator.
        \nX1, X2 are integers. Please do not include title such as "X1" in the output. Output format should be: "X1\nX2\nZ1\nZ2".\n"""
        
        self.evaluation_instruction_2 = """\nLet X1 be the number of mentioned categories of objects in the text prompt.\nLet X2 be the number of categories of objects in X1 that are matched in the image description.\nLet Y1 be the number of color, numerical counting, shape, size, location attributes of each object in X1.\nLet Y2 be the number of attributes in Y1 that are matched in the image description.\nLet Z be the explanation of getting X1,X2,Y1,Y2 within a paragraph less than 6 sentence.\nX1, X2, Y1, Y2 are all integers. Z is a paragraph without separator. Please do not include title such as \"X1\" in the output. Output should be only the value of X1,X2,Y1,Y2,Z with the format \"X1\nX2\nY1\nY2\nZ\".\n"""
        
    def generate_score_with_rationale(self, description, text_prompt, use_rule=False):
        prompt = f"Text Prompt: {text_prompt}\nImage Description: {description}\n{self.evaluation_instruction_1}"
        if self.llm_id in ["gpt-3.5-turbo", "gpt-4"]:
            completion = openai.ChatCompletion.create(
                model=self.llm_id, 
                messages = [
                {"role": "user", "content" : prompt}]
            )
        elif self.llm_id in ["vicuna"]:
            openai.api_base = "http://localhost:8000/v1"
            model = "vicuna-7b-v1.1"
            completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
            )
        else:
            completion = self.completion.create(prompt=prompt, engine=self.llm_id)
        response = completion['choices'][0]['message']['content']
        response = "\n".join(item for item in response.split('\n') if item)
        if use_rule:
            X1,X2,Y1,Y2,rationale = response.split("\n")
            avg_score = ((int(X2)/int(X1))+(int(Y2)/int(Y1)))/2
            return avg_score, rationale
        else:
            overall, error_counting, overall_rationale, error_counting_rationale = response.split("\n")[:4]
            return overall, error_counting, overall_rationale, error_counting_rationale
        
