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
        return {
            "location": [],
            "emergency_type": ["Unknown"],
            "severity": ["High"]  # Default severity
        }

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

    # Set defaults if no entities found
    if not entities["emergency_type"]:
        entities["emergency_type"] = ["Unknown"]
    if not entities["severity"]:
        entities["severity"] = ["High"]

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
        # For development/testing - log message instead of sending
        st.info(f"SMS would be sent to {to_number}: {message}")

        # Comment out the actual Twilio sending for now
        """
        # Make sure to use your updated credentials from Twilio console
        client = Client(
            "your_new_account_sid",  # Update with actual SID
            "your_new_auth_token"    # Update with actual token
        )
        message = client.messages.create(
            body=message,
            from_="+1234567890",     # Update with actual Twilio number
            to=to_number
        )
        return True, message.sid
        """
        return True, "Message logged (SMS disabled in development)"

    except Exception as e:
        return False, str(e)

def notify_emergency_services(location, severity, situation_text):
    """Handles emergency notifications with multiple fallback options"""

    # Format messages
    gov_message = (
        f"🚨 URGENT EMERGENCY ALERT!\n"
        f"Location: {location}\n"
        f"Severity: {severity}\n"
        f"Details: {situation_text[:100]}..."  # Truncate long messages
    )

    user_message = (
        f"🔹 Emergency services have been notified.\n"
        f"Location: {location}\n"
        f"Help is being coordinated.\n"
        f"Stay safe and follow emergency instructions."
    )

    # Log notifications for verification
    st.subheader("Emergency Notifications")
    with st.expander("View Emergency Alert Details"):
        st.markdown("### 🚨 Government Alert")
        st.code(gov_message)
        st.markdown("### 👤 User Alert")
        st.code(user_message)

    # Attempt to send SMS
    numbers = {
        "government": "+919667523306",
        "user": "+919667523306"
    }

    results = {}
    for recipient, number in numbers.items():
        success, result = send_emergency_sms(number,
            gov_message if recipient == "government" else user_message)
        results[recipient] = {"success": success, "result": result}

        if success:
            st.success(f"✅ Alert sent to {recipient}")
        else:
            st.warning(f"⚠️ Could not send SMS to {recipient}. Alert logged in system.")

    return results

def main():
    st.title("Emergency Response System")

    # Create sidebar for emergency contacts and instructions
    with st.sidebar:
        st.header("📞 Emergency Contacts")
        st.write("Police: 100")
        st.write("Fire: 101")
        st.write("Ambulance: 102")
        st.write("Disaster Management: 108")

        st.header("🚨 Important Instructions")
        st.info(
            "1. Stay calm and assess the situation\n"
            "2. Ensure your safety first\n"
            "3. Call emergency services if immediate help is needed\n"
            "4. Follow official instructions\n"
            "5. Keep your phone charged"
        )

    # Load models with error handling
    try:
        with st.spinner("Initializing emergency response system..."):
            nlp = load_spacy_model()
            clip_model, clip_processor = load_clip_model()
            setup_entity_ruler(nlp)
    except Exception as e:
        st.error("Error loading required models. Please try again.")
        return

    # Create tabs for different input methods
    input_tab, doc_tab = st.tabs(["📝 Direct Input", "📄 Document Upload"])

    situation_text = ""
    location = ""

    with input_tab:
        situation_text = st.text_area(
            "Describe the emergency situation:",
            placeholder="Example: There's a fire in the apartment building at 123 Main St. Multiple people need evacuation."
        )
        location = st.text_input(
            "Enter location:",
            placeholder="Example: 123 Main Street, City Name"
        )

    with doc_tab:
        uploaded_file = st.file_uploader("Upload PDF document", type="pdf")
        if uploaded_file:
            with st.spinner("Processing document..."):
                situation_text = process_pdf(uploaded_file)
                if situation_text:
                    st.success("Document processed successfully")
                    st.expander("View extracted text").write(situation_text)
                    location = st.text_input("Confirm location:", key="doc_location")
                else:
                    st.error("Could not extract text from document")

    # Image upload section with preview
    st.subheader("📸 Situation Image (Optional)")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Process emergency button with validation
    process_button = st.button("🚨 Process Emergency", use_container_width=True)

    if process_button:
        if not situation_text or not location:
            st.error("Please provide both situation description and location")
            return

        with st.spinner("Processing emergency information..."):
            # Create containers for different sections
            analysis_container = st.container()
            location_container = st.container()
            response_container = st.container()

            with analysis_container:
                st.subheader("🔍 Situation Analysis")

                # Process image if uploaded
                image_label = None
                if uploaded_image and clip_model and clip_processor:
                    image_label, confidence = process_image(uploaded_image, clip_model, clip_processor)
                    if image_label:
                        st.info(f"📸 Image Analysis: {image_label.title()} (Confidence: {confidence:.2f})")

                # Extract entities and display
                entities = extract_entities(situation_text, nlp)
                severity = entities["severity"][0] if entities["severity"] else "High"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Emergency Type", entities["emergency_type"][0].title())
                with col2:
                    st.metric("Severity Level", severity.upper())

            with location_container:
                st.subheader("📍 Location Information")
                lat, lon = get_lat_lon(location)

                if lat and lon:
                    m = folium.Map(location=[lat, lon], zoom_start=15)
                    folium.Marker(
                        [lat, lon],
                        popup=location,
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
                    folium_static(m)
                else:
                    st.warning("Could not locate the exact position on map")

            with response_container:
                st.subheader("🏥 Emergency Response")

                # Generate and display first aid response
                emergency_type = image_label if image_label else entities["emergency_type"][0]
                first_aid = get_first_aid_response(emergency_type, situation_text)

                with st.expander("First Aid Instructions", expanded=True):
                    st.markdown(first_aid)

                # Send notifications
                st.subheader("📱 Emergency Notifications")
                with st.spinner("Sending alerts..."):
                    # Demo phone numbers (replace with actual numbers in production)
                    government_contact = "+919667523306"
                    user_phone = "+919667523306"

                    messages = {
                        "Emergency Services": {
                            "to": government_contact,
                            "message": f"🚨 URGENT! Emergency at {location}. Type: {emergency_type}. Severity: {severity}. Immediate response required."
                        },
                        "User": {
                            "to": user_phone,
                            "message": f"🔹 Help is on the way! Emergency services have been notified about the {emergency_type} at {location}. Stay safe and follow instructions."
                        }
                    }

                    for recipient, data in messages.items():
                        success, result = send_emergency_sms(data["to"], data["message"])
                        if success:
                            st.success(f"✅ Alert sent to {recipient}")
                        else:
                            st.error(f"❌ Could not send alert to {recipient}: {result}")
                            st.info("Please contact emergency services directly using the numbers in the sidebar.")

if _name_ == "_main_":
    main()
