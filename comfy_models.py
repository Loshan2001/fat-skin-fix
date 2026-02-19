MODEL_LIST = [

    # =======================================================
    # 1. MAIN SKIN CHECKPOINT (CUSTOM)
    # =======================================================
    {
        "url": "https://huggingface.co/alexShangeeth/LastFamedmd2/resolve/main/VXVI_LastFame_DMD2.safetensors",
        "path": "/data/models/checkpoints/epicrealismXL_vxviLastfameDMD2.safetensors",
        "target": "/comfyui/models/checkpoints/epicrealismXL_vxviLastfameDMD2.safetensors"
    },

    # =======================================================
    # 2. SEEDVR2 UPSCALER ✅ FINAL FIXED LOCATION
    # =======================================================
    {
        "url": "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_7b_fp16.safetensors",
        "path": "/data/models/SEEDVR2/seedvr2_ema_7b_fp16.safetensors",
        "target": "/comfyui/models/SEEDVR2/seedvr2_ema_7b_fp16.safetensors"
    },
    {
        "url": "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "path": "/data/models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "target": "/comfyui/models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    },
    {
        "url": "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors",
        "path": "/data/models/SEEDVR2/ema_vae_fp16.safetensors",
        "target": "/comfyui/models/SEEDVR2/ema_vae_fp16.safetensors"
    },

    # =======================================================
    # 3. FACE PARSING
    # =======================================================
    {
        "url": "https://huggingface.co/jonathandinu/face-parsing/resolve/main/model.safetensors",
        "path": "/data/models/face_parsing/model.safetensors",
        "target": "/comfyui/models/face_parsing/model.safetensors"
    },
    {
        "url": "https://huggingface.co/jonathandinu/face-parsing/resolve/main/preprocessor_config.json",
        "path": "/data/models/face_parsing/preprocessor_config.json",
        "target": "/comfyui/models/face_parsing/preprocessor_config.json"
    },
    {
        "url": "https://huggingface.co/jonathandinu/face-parsing/resolve/main/config.json",
        "path": "/data/models/face_parsing/config.json",
        "target": "/comfyui/models/face_parsing/config.json"
    },

    # =======================================================
    # 4. INSIGHTFACE - INSWAPPER
    # =======================================================
    {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "path": "/data/models/insightface/inswapper_128.onnx",
        "target": "/comfyui/models/insightface/inswapper_128.onnx"
    },

    # =======================================================
    # 5. VITMATTE BACKGROUND MATTING
    # =======================================================
    {
        "url": "https://huggingface.co/shiertier/vitmatte/resolve/main/model.safetensors",
        "path": "/data/models/vitmatte/model.safetensors",
        "target": "/comfyui/models/vitmatte/model.safetensors"
    },
    {
        "url": "https://huggingface.co/shiertier/vitmatte/resolve/main/config.json",
        "path": "/data/models/vitmatte/config.json",
        "target": "/comfyui/models/vitmatte/config.json"
    },
    {
        "url": "https://huggingface.co/shiertier/vitmatte/resolve/main/preprocessor_config.json",
        "path": "/data/models/vitmatte/preprocessor_config.json",
        "target": "/comfyui/models/vitmatte/preprocessor_config.json"
    },
    {
        "url": "https://huggingface.co/shiertier/vitmatte/resolve/main/pytorch_model.bin",
        "path": "/data/models/vitmatte/pytorch_model.bin",
        "target": "/comfyui/models/vitmatte/pytorch_model.bin"
    },

    # =======================================================
    # 6. FLORENCE-2 BASE (FULL VISION + LLM SET)
    # =======================================================
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/model.safetensors",
        "path": "/data/models/LLM/Florence-2-base/model.safetensors",
        "target": "/comfyui/models/LLM/Florence-2-base/model.safetensors"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/pytorch_model.bin",
        "path": "/data/models/LLM/Florence-2-base/pytorch_model.bin",
        "target": "/comfyui/models/LLM/Florence-2-base/pytorch_model.bin"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/config.json",
        "path": "/data/models/LLM/Florence-2-base/config.json",
        "target": "/comfyui/models/LLM/Florence-2-base/config.json"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/preprocessor_config.json",
        "path": "/data/models/LLM/Florence-2-base/preprocessor_config.json",
        "target": "/comfyui/models/LLM/Florence-2-base/preprocessor_config.json"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/tokenizer.json",
        "path": "/data/models/LLM/Florence-2-base/tokenizer.json",
        "target": "/comfyui/models/LLM/Florence-2-base/tokenizer.json"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/tokenizer_config.json",
        "path": "/data/models/LLM/Florence-2-base/tokenizer_config.json",
        "target": "/comfyui/models/LLM/Florence-2-base/tokenizer_config.json"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/vocab.json",
        "path": "/data/models/LLM/Florence-2-base/vocab.json",
        "target": "/comfyui/models/LLM/Florence-2-base/vocab.json"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/configuration_florence2.py",
        "path": "/data/models/LLM/Florence-2-base/configuration_florence2.py",
        "target": "/comfyui/models/LLM/Florence-2-base/configuration_florence2.py"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/modeling_florence2.py",
        "path": "/data/models/LLM/Florence-2-base/modeling_florence2.py",
        "target": "/comfyui/models/LLM/Florence-2-base/modeling_florence2.py"
    },
    {
        "url": "https://huggingface.co/microsoft/Florence-2-base/resolve/main/processing_florence2.py",
        "path": "/data/models/LLM/Florence-2-base/processing_florence2.py",
        "target": "/comfyui/models/LLM/Florence-2-base/processing_florence2.py"
    },

    # =======================================================
    # 7. MEDIAPIPE SELFIE SEGMENTATION
    # =======================================================
    {
        "url": "https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite",
        "path": "/data/models/mediapipe/selfie_multiclass_256x256.tflite",
        "target": "/comfyui/models/mediapipe/selfie_multiclass_256x256.tflite"
    },

    # =======================================================
    # 8. YOLOv8 FACE DETECTION (ULTRALYTICS - BBOX)
    # =======================================================
    {
        "url": "https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/ultralytics/bbox/face_yolov8m.pt",
        "path": "/data/models/ultralytics/bbox/face_yolov8m.pt",
        "target": "/comfyui/models/ultralytics/bbox/face_yolov8m.pt"
    }
]
