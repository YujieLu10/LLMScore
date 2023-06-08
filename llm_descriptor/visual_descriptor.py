import openai
class VisualDescriptor:
    def __init__(self, api_key, llm_id="gpt-4"):
        openai.api_key = api_key
        self.llm_id = llm_id
    
    def generate_multi_granualrity_description(self, local_description, global_description, width, height):
        prompt = f"\nGlobal Description: {local_description}\nLocal Description: {global_description}\nThe image resolution is:{width}X{height}\nBased on the global description, local description of the generated image, please generate a detailed image description (only one paragraph with no more than 10 sentences) that describe the color, spatial position, shape, size, material of each object, and relationship among objects. The location of the object should be in natural language format instead of numerical coordinates.\n"
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
            completion = openai.ChatCompletion.create(
                model=self.llm_id, 
                messages = [
                {"role": "user", "content" : prompt}]
            )
        return completion['choices'][0]['message']['content'].strip().replace("\n", " ")
