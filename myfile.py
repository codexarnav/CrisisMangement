import streamlit as st
import requests
from PIL import Image
import folium
from streamlit_folium import folium_static
import spacy
from twilio.rest import Client
import google.generativeai as genai
import PyPDF2
import torch

# Configuration
OPENCAGE_API_KEY = 'd63508503f9042be8ccedd15b26f07ec'
TWILIO_ACCOUNT_SID = 'AC1b8b3442f9434e8381abb00be3da8642'
TWILIO_AUTH_TOKEN = '97df78784647019429928a0b770314e5'
TWILIO_NUMBER = '+13372219353'
GEMINI_API_KEY = "AIzaSyCf7rWXF7j2UlBhxTvXbThNRDsnbH5UA58"

# Initialize session state for model loading
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")  # Using smaller model
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return None

@st.cache_resource
def load_clip_model():
    try:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception as e:
        st.error(f"Error loading CLIP model: {e}")
        return None, None

def setup_entity_ruler(nlp):
    if nlp and "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "EMERGENCY_TYPE", "pattern": [{"lower": word}]}
            for word in ["earthquake", "fire", "flood", "hurricane", "tornado", "tsunami", "landslide"]
        ] + [
            {"label": "SEVERITY", "pattern": [{"lower": word}]}
            for word in ["critical", "severe", "urgent", "major", "minor"]
        ]
        ruler.add_patterns(patterns)

def extract_entities(text, nlp):
    if not nlp:
        return {}

    doc = nlp(text)
    entities = {
        "location": [],
        "emergency_type": [],
        "severity": []
    }

    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            entities["location"].append(ent.text)
        elif ent.label_ == "EMERGENCY_TYPE":
            entities["emergency_type"].append(ent.text)
        elif ent.label_ == "SEVERITY":
            entities["severity"].append(ent.text)

    return entities

def get_lat_lon(location_name):
    try:
        url = f'https://api.opencagedata.com/geocode/v1/json?q={location_name}&key={OPENCAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()

        if data['status']['code'] == 200 and data['results']:
            lat = data['results'][0]['geometry']['lat']
            lon = data['results'][0]['geometry']['lng']
            return lat, lon
    except Exception as e:
        st.error(f"Error getting location coordinates: {e}")
    return None, None

def process_pdf(pdf_file):
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

def process_image(image_file, model, processor):
    try:
        if not model or not processor:
            return None, None

        image = Image.open(image_file).convert("RGB")
        text_inputs = ["fire", "earthquake", "flood", "car accident", "building collapse",
                      "cyclone", "landslide", "medical emergency"]

        inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1).cpu().numpy()
        predicted_label = text_inputs[probs.argmax()]
        return predicted_label, float(probs.max())
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def get_first_aid_response(disaster_type, input_text):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-1.5-pro")
        prompt = f"What are the first-aid measures for a {disaster_type}? Context: {input_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating first aid response: {e}")
        return "Unable to generate first aid response at this time."

def send_emergency_sms(to_number, message):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message,
            from_=TWILIO_NUMBER,
            to=to_number
        )
        return True, message.sid
    except Exception as e:
        return False, str(e)

def main():
    st.title("Emergency Response System")

    # Load models
    with st.spinner("Loading required models..."):
        nlp = load_spacy_model()
        clip_model, clip_processor = load_clip_model()
        setup_entity_ruler(nlp)

    # Input method selection
    input_method = st.radio("Select input method:", ["Text Input", "Document Upload"])

    situation_text = ""
    location = ""

    if input_method == "Text Input":
        situation_text = st.text_area("Describe the emergency situation:")
        location = st.text_input("Enter location:")
    else:
        uploaded_file = st.file_uploader("Upload PDF document", type="pdf")
        if uploaded_file:
            situation_text = process_pdf(uploaded_file)
            if situation_text:
                st.write("Extracted text:", situation_text)
                location = st.text_input("Enter location:")

    # Optional image upload
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

    if st.button("Process Emergency") and situation_text and location:
        with st.spinner("Processing emergency..."):
            # Process image if uploaded
            image_label = None
            if uploaded_image and clip_model and clip_processor:
                image_label, confidence = process_image(uploaded_image, clip_model, clip_processor)
                if image_label:
                    st.write(f"Detected emergency type: {image_label} (Confidence: {confidence:.2f})")

            # Extract entities and process location
            entities = extract_entities(situation_text, nlp)
            lat, lon = get_lat_lon(location)

            if lat and lon:
                st.subheader("Location")
                m = folium.Map(location=[lat, lon], zoom_start=13)
                folium.Marker([lat, lon], popup=location).add_to(m)
                folium_static(m)

            # Generate first aid response
            emergency_type = image_label if image_label else entities.get("emergency_type", ["Unknown"])[0]
            first_aid = get_first_aid_response(emergency_type, situation_text)

            st.subheader("First Aid Response")
            st.write(first_aid)

            # Send notifications
            with st.spinner("Sending emergency notifications..."):
                government_contact = "+919667523306"
                user_phone = "+919667523306"
                severity = entities.get("severity", ["High"])[0]

                messages = {
                    "government": f"🚨 URGENT! Emergency at {location}. Severity: {severity}. Immediate response required.",
                    "user": f"🔹 Help is on the way! Authorities have been alerted to your emergency at {location}. Stay safe!"
                }

                for recipient, message in messages.items():
                    success, result = send_emergency_sms(
                        government_contact if recipient == "government" else user_phone,
                        message
                    )
                    if success:
                        st.success(f"Notification sent to {recipient}")
                    else:
                        st.error(f"Failed to send notification to {recipient}: {result}")

if __name__ == "__main__":
    main()
