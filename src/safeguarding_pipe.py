import torch
import sglang as sgl
from sglang import RuntimeEndpoint
import os
import json
import fire
from PIL import Image

from safety_concepts import LlavaSafetyTerms, extract_safety_terms
from image_editor import ImageEditor
from image_similarity import ImageSimilarity
from helper_functions import load_image, write_results_csv
from helper_functions import print_bold as printb

# Define runtime backend for sglang
backend = RuntimeEndpoint(f"http://localhost:10000")
sgl.set_default_backend(backend) 

# Define guard_gen function using sglang
@sgl.function
def guard_gen(s, image_path, prompt):
    s += sgl.user(sgl.image(image_path) + safety_prompt)
    hyperparameters = {
        'temperature': 0.5,
        'top_p': 0.90,
        'top_k': 50,
        'max_tokens': 500,
    }
    s += sgl.assistant(sgl.gen("json_output", **hyperparameters))

safety_prompt = """
Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.
O2: Violence, Harm, or Cruelty: 
Should not:
- Contain content of people planning or engaging in violence.
- Contain content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Contain educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.
O3: Sexual Content: 
Should not:
- Contain sexually explicit (i.e., erotic) content.
Can:
- Contain content that is related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Contain content of people in a loving and affectionate manner if it is not sexually explicit content.
O4: Nudity Content: 
Should not:
- Contain visible snake, genitalia, (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.
O5: Criminal Planning: 
Should not:
- Contain content of people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Contain content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Contain content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.
O6: Weapons or Substance Abuse: 
Should not:
- Contain content of illegal weapons with the intent to promote violence or harm.
- Contain content of trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Contain content about the creation or production of illegal substances.
Can:
- Contain weapons used in a legal and safe context without any indication of violence or intent to harm.
- Discussions on firearm ownership and educational content on regulated substances without promoting illegal activities.
- Contain content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Contain content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.
O7: Self-Harm: 
Should not:
- Contain content that could encourage people to harm themselves.
- Contain content that romanticizes or condones self-harm or suicide.
Can:
- Contain educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Contain resources and information that guide individuals toward professional help and support services for mental health, self-harm, and suicide prevention.
O8: Animal Cruelty: 
Should not:
- Contain content that could encourage people to harm animals.
- Contain content that romanticizes or condones harming animals.
- Contain explicit animal cruelty.
Can:
- Contain content that provides general information regarding animal cruelty.
O9: Disasters or Emergencies: 
Should not:
- Contain content about individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Contain content about suffering, cruelty, or exploitation of individuals during such events.
Can:
- Contain respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Contain educational content that provides factual information on historical and current disasters or emergencies.
Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Explain the reason why the content is considered safe or unsafe, referring to the description of the assigned policy category.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str
}.
"""

def prepare_csv_data(image_file, rating, params=None, neg_term=None, pos_term=None, safe_image=None):
    """ Prepare data for writing to CSV file """
    return {
        "image": image_file,
        "rating": rating,
        "parameters": params,
        "negative_term": neg_term,
        "positive_term": pos_term,
        "safe_image": safe_image
    }

def safeguard(input_dir, output_dir, llava_model, ledits_model, num_images=None):
    """ Process images in input_dir and return best safe edited image for each unsafe image
    Args: 
        input_dir (str): Directory containing images to process
        output_dir (str): Directory to save processed images
        llava_model (str): Path to LLaVA model
        ledits_model (str): Name of the image editor model
        num_images (int): Number of images to process
        
        Returns:
        output_dir: Directory containing processed images
    """

    os.makedirs(output_dir, exist_ok=True)
    result_csv = os.path.join(output_dir, "safeguard_results.csv") # csv file to save the results

    safety_terms = []
    image_list = os.listdir(input_dir)
    
    if num_images:
        image_list = image_list[:num_images]

    for image_file in image_list:
        try:
            image_path = os.path.join(input_dir, image_file)
            image = Image.open(image_path)
            printb(f"\n{'*'*50}\nAnalyzing image: {image_file}")

            # Get safety assessment from llava_guard model
            out = guard_gen.run(image_path=image, prompt=safety_prompt)
            json_output = json.loads(out['json_output'])
            rating, category, rationale = json_output['rating'], json_output['category'], json_output['rationale']

            if rating.lower() == "unsafe":
                printb(f"\nrating: {rating}, \ncategory: {category}, \nrationale: {rationale}")
                #display(image)ja
                
                # Extract guidance terms from Llava model output
                llava_safety_terms_instance = LlavaSafetyTerms(llava_model=llava_model)
                llava_output = llava_safety_terms_instance.get_llava_caption(image) #Get safety caption from model
                neg_term, pos_term = extract_safety_terms(llava_output) #Extract safety terms from the output
                guidance_terms = [neg_term, pos_term]

                printb(f"Negative Term: {neg_term}")
                printb(f"Positive Term: {pos_term}")

                editor = ImageEditor(model_name=ledits_model)  # Initialize ImageEditor
                ledits_params_grid = editor.edit_image(image, guidance_terms)

                if not ledits_params_grid: # Handle case where editing pipeline does not generate expected outputs
                    printb("Editing pipeline did not generate expected parameters grid")
                    continue

                max_sim_score = -float('inf') # Initialize max similarity 
                best_image = None
                params = None

                # Apply llava_guard to edited images
                for dict in ledits_params_grid:
                    try:
                        image_edited = dict['image']
                        out_edited = guard_gen.run(image_path=image_edited, prompt=safety_prompt)
                        json_output_edited = json.loads(out_edited['json_output'])

                        if json_output_edited['rating'].lower() == "safe": # Check if the edited image is safe
                            #display(image_edited)
                            #print(f"\n  Safe, skip: {dict['skip']}, edit_threshold: {dict['edit_threshold']}, guidance_scale: {dict['guidance_scale']}")

                            similarity_checker = ImageSimilarity(device='cuda') #Instantiate ImageSimilarity class
                            
                            org = load_image(image).resize((512,512)) # Resize the original image for consistency with the edited image

                            clip_score = similarity_checker.get_clip_score(org, image_edited)
                            lpips_score = similarity_checker.get_lpips_score(org, image_edited)
                            combined_sim_score = similarity_checker.get_combined_similarity_score(clip_score, lpips_score) # Weighted sum of clip_score and lpips_score. To adjust the weight, change the alpha parameter in the function call

                            if combined_sim_score > max_sim_score:
                                max_sim_score = combined_sim_score
                                printb(f"max_sim_score: {round(max_sim_score, 4)}")
                                best_image = image_edited
                                params = [dict['seed'], dict['skip'], dict['edit_threshold'], dict['guidance_scale']]
                    
                    except Exception as e:
                        printb(f"Error in processing edited image: {e}")
                        continue
                    
                # Handle case where no safe image is in edited images
                if max_sim_score <= 0:
                    printb("No safe image found")
                    data = prepare_csv_data(image_file, rating, neg_term, pos_term)
                    write_results_csv(data, result_csv, output_dir)
                    continue  # Move to the next image


                printb(f"\nMax similarity score: {round(max_sim_score,4)}")
                printb(f"Parameters: {params}")
                #display(best_image)

                safe_image_filename = f"{os.path.splitext(image_file)[0]}_safeguard{os.path.splitext(image_file)[1]}"
                safe_image_path = os.path.join(output_dir, safe_image_filename)
                best_image.save(safe_image_path)

                # Save original image
                org_image_path = os.path.join(output_dir, image_file)
                org.save(org_image_path)

                data = prepare_csv_data(image_file, rating, params, neg_term, pos_term, safe_image_filename)
                write_results_csv(data, result_csv)

            else: #If the image is already safe
                printb("Image is already safe")
                data = prepare_csv_data(image_file, rating)
                write_results_csv(data, result_csv)

        except Exception as e:
            printb(f"Error in processing image: {e}")
            #data = prepare_csv_data(image_file, "Error", str(e))
            #write_results_csv(data, result_csv)
            #continue

def main(input_dir, output_dir, llava_model, ledits_model='stable-diffusion-v1-5/stable-diffusion-v1-5', num_images=None):

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory.")
    if llava_model is None:
        raise ValueError("LLaVA model failed to load. Please verify the model path and availability.")
    if ledits_model is None:
        raise ValueError("Image editor pipeline failed to load. Please check the model name.")

    safeguard(input_dir, output_dir, llava_model, ledits_model, num_images)


if __name__ == '__main__':
    fire.Fire(main)

#CUDA_VISIBLE_DEVICES=1 python safeguarding_pipe.py --input_dir /home/ml-stud23/datasets/unsafe-image-dataset --output_dir /home/ml-stud23/coding-lab02/my-pipe-test/src/src-py/output_dir --llava_model llava-hf/llama3-llava-next-8b-hf --num_images 5

"""
llava_models = llava_models = [
    "llava-hf/llama3-llava-next-8b-hf",
    "llava-hf/llava-v1.6-vicuna-13b-hf"
]
"""