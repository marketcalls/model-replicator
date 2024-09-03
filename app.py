import os
import requests
import replicate
import re
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import logging

app = Flask(__name__)
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL")
LORA_URL = os.getenv("LORA_URL")
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL")

logging.basicConfig(level=logging.INFO)

def replace_words_with_trigger(text, words_to_replace, trigger_word):
    # Create a regular expression pattern
    pattern = re.compile(r'\b(' + '|'.join(words_to_replace) + r')\b', re.IGNORECASE)
    
    # Replace the matched words with the trigger word
    return pattern.sub(trigger_word, text)

def remove_markdown(text):
    # Remove headers
    text = re.sub(r'#{1,6}\s?', '', text)
    # Remove bold
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove italic
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove bullet points
    text = re.sub(r'^\s*[-*+]\s', '', text, flags=re.MULTILINE)
    # Remove numbered lists
    text = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    # Remove links
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate_description', methods=['POST'])
def generate_description():
    control_image_url = request.form.get('control_image_url', '')
    words_to_replace = request.form.get('words_to_replace', 'woman,women,girl,lady,female').split(',')
    trigger_word = request.form.get('trigger_word', 'VIDHYAASREE')
    
    if not control_image_url:
        return jsonify({'error': 'No Control Image URL provided'})
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image in great detail. Describe the following aspects:\n\n1. Subject: Appearance, pose, and expressions.\n2. Clothing: Colors, styles, and types of garments worn.\n3. Setting: Location, background elements, and overall atmosphere.\n4. Lighting: Quality, direction, and mood created by the lighting.\n5. Composition: Camera angle, framing, and focal points.\n6. Technical details: Type of camera, lens, and any post-processing effects.\n7. Emotions and mood: The overall feeling conveyed by the image.\n8. Actions and interactions: What the subject is doing and how they interact with the environment.\n9. Notable details: Any unique or standout elements in the image.\n\nProvide a comprehensive and vivid description that captures both the visual elements and the essence of the photograph."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": control_image_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 750
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response.json()

    if 'choices' in result and len(result['choices']) > 0:
        description = result['choices'][0]['message']['content']
        
        # Remove markdown formatting
        clean_description = remove_markdown(description)
        
        # Replace words with trigger word
        modified_description = replace_words_with_trigger(clean_description, words_to_replace, trigger_word)
        
        return jsonify({
            'modified_description': modified_description,
        })
    else:
        return jsonify({'error': 'Failed to get a description from the API'})

@app.route('/generate_image', methods=['POST'])
def generate_image():
    control_image_url = request.form.get('control_image_url', '')
    negative_prompt = request.form.get('negative_prompt', 'low quality, ugly, distorted, artefacts, necklace, text')
    control_type = request.form.get('control_type', 'depth')
    model_description = request.form.get('model_description', '')
    
    if not control_image_url or not model_description:
        return jsonify({'error': 'Missing required parameters: control_image_url or model_description'})
    
    # Save input parameters in a separate variable
    input_params = {
        "steps": 28,
        "prompt": model_description,
        "lora_url": LORA_URL,
        "control_type": control_type,
        "control_image": control_image_url,
        "lora_strength": 1,
        "output_format": "png",
        "guidance_scale": 2.5,
        "output_quality": 100,
        "negative_prompt": negative_prompt,
        "control_strength": 0.65,
        "depth_preprocessor": "DepthAnything",
        "soft_edge_preprocessor": "HED",
        "image_to_image_strength": 0.13,
        "return_preprocessed_image": False
    }
    
    # Print input parameters for debugging
    print("Input parameters for replicate.run:")
    for key, value in input_params.items():
        print(f"{key}: {value}")
    
    try:
        # Split the REPLICATE_MODEL into name and version
        model_name, model_version = REPLICATE_MODEL.split(":")
        
        # Get the model
        model = replicate.models.get(model_name)
        
        # Get the specific version of the model
        version = model.versions.get(model_version)
        
        # Create a prediction
        prediction = replicate.predictions.create(
            version=version,
            input=input_params
        )
        
        # Wait for the prediction to complete
        while prediction.status not in ["succeeded", "failed", "canceled"]:
            prediction.reload()
            time.sleep(1)
        
        if prediction.status == "succeeded":
            output = prediction.output
            logging.info(f"Replicate output: {output}")
            
            generated_image_url = output[0] if isinstance(output, list) and output else None
            
            if generated_image_url:
                return jsonify({
                    'generated_image_url': generated_image_url
                })
            else:
                return jsonify({'error': 'Failed to generate image'})
        else:
            return jsonify({'error': f'Prediction failed with status: {prediction.status}'})
    
    except Exception as e:
        logging.error(f"Error during image generation: {str(e)}")
        return jsonify({'error': f'Failed to generate image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)