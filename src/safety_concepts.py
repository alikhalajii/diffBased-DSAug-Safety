import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import os
import json

msg_system = """
As an unsafe-content detector, your primary responsibility is to accurately assess and describe offensive or inappropriate content in images based on the following detailed policy guidelines.

Policy:
1.Hate Speech & Discrimination:
Content expressing hateful, derogatory, or demeaning sentiments against individuals or groups based on sensitive personal characteristics (e.g., race, color, religion, national origin, sexual orientation, gender, gender identity, disability).
Content encouraging discrimination, harassment, or violence against individuals or groups based on these characteristics.

2.Sexual:
Erotic content or imagery depicting explicit sexual activities.
Content showing visible genitalia, including exposed breasts, nipples, vaginas, vulvas, or penises.
Explicit nudity, intimate touching, sexual fetishes, or depictions involving sexual tools.


3.Criminal Activities:
Content depicting or encouraging criminal activities (e.g., theft, kidnapping, financial crimes, vandalism, smuggling).
Content promoting illegal weaponry, trafficking, or consumption and use of illegal drugs.
Content encouraging the production of illegal substances or behaviors that condone criminal activities.

4.Substance Abuse:
Content promoting the use of tobacco, alcohol, syringes, overdose, injection, or cannabis.
Content encouraging substance abuse or presenting it in a glorifying manner.

5.Animal Cruelty:
Content depicting or encouraging harm to animals, including:
Animal Abuse, Animal Beating, Neglect, Animal Torture, Exploitation, Dog Fighting, Cockfighting, Animal Trapping, Animal Skinning,
Poaching, Animal Starvation, Animal Experimentation, Animal Chaining, Killing, Hunting.

6.Violence & Harm:
Content depicting or encouraging violence or cruelty, including:
fighting, use of weapons, suicide, hanging, cutting, wounds, blood, guillotine, execution, torture, or harm to others.
Content showing individuals planning or engaging in violent activities.
Explicit depiction or encouragement of self-harm or suicide.

7.Sensitive:
Content about deceased individuals or those who have suffered harm due to disasters or emergencies, including:
Death, Corpses, Suffering, Grief, Mourning, Injury, Exploitation, Cruelty, Destruction, Harm, Disaster Aftermath, War Casualties, Tragedy, Starvation, Homelessness, Trauma, Burn Victims, Drowning Victims.

Assessment Instructions:
1. Rationale: 
Describe why the content in the image is considered offensive or inappropriate according to the policy. Ensure the explanation addresses the exact element of the image that violates the policy.

2. Negative Concept:
Identify the most offensive part of the image using one or two words that capture the core issue. These words should accurately reflect the most problematic aspect of the image, based on the relevant policy details. Use diverse vocabulary that specifically emphasizes the identified issue.


3. Positive Concept: 
Suggest a specific counteracting concept based on the image context, using a maximum of two words, that directly addresses the detected offensive element (e.g., "clothed" or "dressed" for nudity).


Only output the assesment in the JSON format provided below:

{
    "rationale": "Rationale",
    "negative_concept": "Negative Concept",
    "positive_concept": "Positive Concept"
}
"""

msg_user = [
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": msg_system
      },
      {
        "type": "image",
        "url": "image_path"
      }
    ]
  },
  {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "Evaluation in JSON format."
      }
    ]
  }
]

# Define the LlavaSafetyTerms class 
class LlavaSafetyTerms:
    def __init__(self, llava_model):
        self.llava_model_id = llava_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_data = self.load_llava_model()

    def load_llava_model(self):
        """ Load the specified LlavaNext model """
        print(f"Loading model {self.llava_model_id}...")
        processor = LlavaNextProcessor.from_pretrained(self.llava_model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.llava_model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.to(self.device)
        return {"processor": processor, "model": model}

    def get_llava_caption(self, image):
        """ Generate a caption for an image using the loaded LlavaNext model """
        processor = self.model_data['processor']
        model = self.model_data['model']
        
        conversation = msg_user

        # Apply processor template and generate
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        output = model.generate(**inputs, max_new_tokens=500)
        conversation_answer = processor.decode(output[0], skip_special_tokens=True)
        
        return conversation_answer


def extract_safety_terms(text):
    """ Extract safety terms from the model's output """
    # Find last JSON object in the text
    try:
        json_start_index = max(index for index, char in enumerate(text) if char in "{[")
        json_end_index = text.rfind('}') + 1
        json_string = text[json_start_index:json_end_index]
    except ValueError:
        raise(f'JSON object does not match the expected format. {ValueError}')
        return None, None

    # Extract negative and positive concepts from the JSON object
    try:
        json_dict = json.loads(json_string)
        neg_term = json_dict["negative_concept"]
        pos_term = json_dict["positive_concept"]
        return neg_term, pos_term
    except json.JSONDecodeError as error:
        raise(f'JSON keys cannot be extracted. {error}')
        return None, None


