from neurosity import neurosity_sdk
from dotenv import load_dotenv
import os

load_dotenv()

neurosity = neurosity_sdk({
    "device_id": os.getenv("NEUROSITY_DEVICE_ID"),
})

neurosity.login({
    "email": os.getenv("NEUROSITY_EMAIL"),
    "password": os.getenv("NEUROSITY_PASSWORD")
})

def callback(data):
    print("data", data)

unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)