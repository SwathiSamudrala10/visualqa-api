import streamlit as st
from PIL import Image, ImageOps
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

# Set page layout to wide
st.set_page_config(layout="wide")

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image, text):
    try:
        # Load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")

        # Prepare inputs
        encoding = processor(img, text, return_tensors="pt")

        # Forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return answer

    except Exception as e:
        return str(e)

def _image_is_gif(pil_image):
    return bool(pil_image.format == "GIF") if pil_image.format else False

def _validate_image_format_string(image_data, output_format):
    if output_format.lower() not in ["jpeg", "png"]:
        raise ValueError(
            "Invalid output format. Please choose 'jpeg' or 'png'."
        )
    return output_format.lower()

def image_to_url(image_data, output_format):
    image_format = _validate_image_format_string(image_data, output_format)
    pil_image = Image.open(BytesIO(image_data))
    if _image_is_gif(pil_image):
        pil_image = pil_image.convert("RGBA")
    st_image = ImageOps.exif_transpose(pil_image)
    image_data = BytesIO()
    st_image.save(image_data, format=image_format.upper())
    return image_data

# Set up the Streamlit app
st.title("Visual Question Answering")
st.write("Upload an image and enter a question to get an answer.")

# Create columns for image upload and input fields
col1, col2 = st.columns(2)

# Image upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)

# Question input
with col2:
    question = st.text_input("Question")

    # Process the image and question when both are provided
    if uploaded_file and question is not None:
        if st.button("Ask Question"):
            image = Image.open(uploaded_file)
            image_byte_array = BytesIO()
            image.save(image_byte_array, format='JPEG')
            image_bytes = image_byte_array.getvalue()

            # Get the answer
            answer = get_answer(image_bytes, question)

            # Display the answer
            st.success("Answer: " + answer)
