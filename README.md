# Model Replicator

Model Replicator is an Flux Lora+Controlnet based image generation tool that allows you to generate AI-generated images based on a control image and a description. It uses the OpenAI GPT-4 Vision API and the Replicate API to generate images.

## Features

- Generate image descriptions from a control image URL
- Edit and refine the generated description
- Create AI-generated images based on the modified description
- Customizable negative prompts and control types
- Side-by-side comparison of control and generated images

## Prerequisites

- Python 3.7+
- Flask
- Replicate API key
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/marketcalls/model-replicator.git
   cd model-replicator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Copy the `.env.sample` file to `.env`:
     ```
     cp .env.sample .env
     ```
   - Open the `.env` file and replace the placeholder values with your actual API keys and settings:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     OPENAI_MODEL=gpt-4-vision-preview
     REPLICATE_API_TOKEN=your_replicate_api_token_here
     LORA_URL=your_lora_url_here
     REPLICATE_MODEL=your_replicate_model_here
     ```

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Enter the control image URL and adjust the parameters as needed

4. Generate and edit the image description

5. Generate the final image based on the modified description

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- OpenAI for providing the GPT-4 Vision API
- Replicate for the image generation model
- Flask for the web framework