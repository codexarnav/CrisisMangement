# Emergency Response System

## Overview

The **Emergency Response System** is a web-based application built using Streamlit that allows users to report emergency situations by providing various inputs such as text, PDF documents, and images. The system processes the inputs, extracts important details like location and emergency type, provides a first-aid response, and sends SMS notifications to the government and the user. The system also visualizes the location of the emergency on a map.

### Key Features:
- Text input for describing the emergency situation and location.
- PDF document upload for automatic text extraction and summarization.
- Optional image upload for automatic emergency type detection.
- Entity extraction using spaCy to identify key emergency-related entities (e.g., emergency type, severity, victim condition).
- Map display showing the extracted location.
- SMS notifications to government authorities and the user, with real-time status updates.
- First-aid response generated based on the situation and emergency type.

## Project Flow

1. **User Input Options:**
    - **Option 1**: Enter the situation and location as text.
    - **Option 2**: Upload a PDF document that contains the emergency details (drag-and-drop).
   
2. **Optional Image Input:**
    - Upload an image (optional) which will be processed to detect the emergency type using CLIP (Contrastive Language-Image Pre-Training).

3. **Processing the Inputs:**
    - If the user uploads a **PDF**, the system will extract the text, summarize it, and identify key entities using spaCy's entity recognition.
    - If the **text-based situation** is entered, it will be processed directly.
    - If an **image** is uploaded, the system will identify the emergency type and label it.

4. **Generate Immediate Response:**
    - Based on the input data (text and image), the system will provide an immediate response, including a first-aid guide based on the detected emergency type.

5. **Location Mapping:**
    - The location is extracted and displayed on an interactive map using **Folium**.

6. **Send SMS Notifications:**
    - Two SMS messages are sent:
        1. **To Government Authorities**: Alerts them to the emergency, with location details and severity.
        2. **To the User**: Confirms that the authorities have been notified and provides reassurance.

## Technologies Used

- **Streamlit**: Framework for building the web-based interface.
- **spaCy**: For named entity recognition (NER) to extract emergency-related entities.
- **BART**: For text summarization of PDF documents.
- **CLIP (OpenAI)**: For processing images and identifying emergency-related content.
- **Folium**: For displaying the location of the emergency on a map.
- **Twilio**: For sending SMS notifications.
- **OpenCage API**: For geocoding the location (latitude and longitude).
- **Google Gemini**: For generating first-aid responses based on the disaster type.

## Setup Instructions

### Prerequisites
- Python 3.x
- Twilio Account (for SMS functionality)
- OpenAI Account (for CLIP model and Gemini)
- OpenCage API Key (for geocoding)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/emergency-response-system.git
    cd emergency-response-system
    ```

2. **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

   The required dependencies are:
   - `streamlit`
   - `requests`
   - `transformers`
   - `spacy`
   - `torch`
   - `Pillow`
   - `google-generativeai`
   - `twilio`
   - `folium`
   - `streamlit_folium`
   - `PyPDF2`

3. **Configure API Keys:**
    - Update the Twilio configuration with your `account_sid`, `auth_token`, and `twilio_number`.
    - Set up your OpenAI API key for CLIP processing and Google Gemini.
    - Add your OpenCage API key for geocoding.

4. **Run the application:**
    ```bash
    streamlit run app.py
    ```

### Usage

1. Open the application in your browser.
2. Choose the input method (either **Text Input** or **PDF Upload**).
3. Optionally, upload an image to help identify the emergency.
4. Click the **"Process Emergency"** button.
5. The system will process the input, display a map with the location, and provide a first-aid response.
6. The system will also send SMS notifications to both the government and the user.

## Code Structure

- **`app.py`**: Main Streamlit app file that controls the flow of the application.
- **`models`**: Directory containing pre-trained models for text summarization, image processing, and first-aid responses.
- **`requirements.txt`**: A list of Python dependencies required for the project.

## Example

### Text Input:
1. **Situation**: "There has been a severe earthquake in the city. Many buildings have collapsed."
2. **Location**: "Los Angeles, California"

### Image Input (optional):
- Upload an image showing collapsed buildings or emergency response situations.

The system will:
1. Parse and summarize the situation.
2. Identify key entities like the emergency type and severity.
3. Display the location on the map.
4. Generate a first-aid response for the emergency.
5. Send SMS notifications to the authorities and the user.

## SMS Notification Example:

1. **To Government**:
    ```
    🚨 URGENT! Emergency at Los Angeles, California. Severity: Severe. Immediate response required.
    ```

2. **To User**:
    ```
    🔹 Help is on the way! Authorities have been alerted to your emergency at Los Angeles, California (Severity: Severe). Stay safe!
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Twilio** for SMS functionality.
- **OpenAI CLIP** for image processing.
- **spaCy** for NLP and entity recognition.
- **Folium** for interactive maps.
- **Google Gemini** for generating first-aid responses.
