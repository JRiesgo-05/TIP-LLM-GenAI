# Chainlit with Google Gemini Integration

This directory contains sample applications demonstrating how to use Chainlit with Google's Gemini API.

## Getting Started

1. **Setup your environment**:
   - Create a `.env` file in the project root with your Google API key:
     ```
     GOOGLE_GENAI_API_KEY=your_api_key_here
     ```

2. **Run the sample application**:
   ```bash
   uv chainlit run chainlit_activity/sample_chainlit_app.py
   ```

3. **Interact with the application**:
   - Open your browser and go to http://localhost:8000
   - Chat with the AI using text
   - Upload images for analysis

## Key Features

- **Text-based Chat**: Ask questions and get responses from Google's Gemini 2.0 Flash model
- **Image Analysis**: Upload images and ask the model to describe or analyze them
- **Customizable System Prompts**: Modify the AI's behavior through the settings panel

## Learning Exercises

1. Try different types of prompts to see how they affect the model's responses
2. Upload various images and see how well the model can analyze them
3. Modify the system prompt to change the AI's behavior
4. Extend the application with new features like audio processing

## Troubleshooting

- If you encounter an error about the API key, make sure your `.env` file is properly set up
- If the application fails to start, ensure you have all dependencies installed
- For any other issues, check the console output for error messages
