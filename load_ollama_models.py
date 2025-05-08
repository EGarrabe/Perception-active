import os
import json

def get_ollama_blob_path(model_name_tag: str, layer_media_type: str) -> str | None:
    """ Finds the full path to an Ollama blob """
    layer_media_type = "application/vnd.ollama.image." + layer_media_type
    user_home = os.path.expanduser("~")
    OLLAMA_MODELS_PATH = os.path.join(user_home, ".ollama", "models")
    manifest_dir = os.path.join(OLLAMA_MODELS_PATH, "manifests", "registry.ollama.ai", "library")

    parts = model_name_tag.split(':', 1)
    model_name = parts[0]
    model_tag = parts[1] if len(parts) > 1 else 'latest'

    manifest_path = os.path.join(manifest_dir, model_name, model_tag)
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        return None

    try:
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
    except Exception as e:
        print(f"Error reading manifest file {manifest_path}: {e}")
        return None

    digest = None
    for layer in manifest_data.get("layers", []):
        if layer.get("mediaType") == layer_media_type:
            digest = layer.get("digest")
            break

    if not digest:
        print(f"Error: Layer type '{layer_media_type}' not found in manifest for {model_name_tag}")
        return None

    blob_filename = digest.replace("sha256:", "sha256-")
    blob_path = os.path.join(OLLAMA_MODELS_PATH, "blobs", blob_filename)

    if not os.path.exists(blob_path):
        print(f"Error: Blob file not found at {blob_path} (derived from digest {digest}). Ensure ollama is running (ollama list)")
        return None

    return blob_path