import threading,extract_features

def mainTextCode(code):

    thread = threading.Thread(target=extract_features.monitor_file, args=(code,),daemon=True)
    thread.start()
