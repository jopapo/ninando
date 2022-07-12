import extract_features
import traceback

#   This is the code to run Text functions...
def mainTextCode(code):
    try:
        result = extract_features.extract_features(code)
        print(result)
    except Exception:
        traceback.print_exc()
