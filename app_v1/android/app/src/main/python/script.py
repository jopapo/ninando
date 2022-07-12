import extract_features
import traceback

#   This is the code to run Text functions...
def mainTextCode(code):
    try:
        extract_features.extract_features(code)
    except Exception:
        traceback.print_exc()
