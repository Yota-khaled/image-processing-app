"""
UI Components and styling for the Image Processing Application
Contains CSS styling and UI helper functions
"""



# Gradio CSS used by app_gradio.py
GRADIO_CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
}
.image-row {
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
    gap: 20px !important;
    margin-bottom: 20px !important;
}
.image-col {
    flex: 1 !important;
    min-width: 0 !important;
}
.image-container {
    border: 2px solid #4A70A9;
    border-radius: 10px;
    padding: 10px;
    background: #1E2B3A;
}
.header {
    text-align: center;
    background: linear-gradient(135deg, #4A70A9 0%, #2A4A7A 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.large-image {
    min-height: 400px !important;
}

.processed-image-gap {
    margin-top: 50px !important;
}
"""


