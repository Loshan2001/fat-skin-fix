WORKFLOW_JSON={
  "input": {
    "uid": "testUid",
    "customNodes": [],
    "customModels": [],
    "images": [
      {
        "name": "de5c01cc.png",
        "image": "base64.."
      }
    ],
    "workflow": {
      "476": {
        "inputs": {
          "samples": [
            "530",
            0
          ],
          "mask": [
            "533",
            1
          ]
        },
        "class_type": "SetLatentNoiseMask",
        "_meta": {
          "title": "Set Latent Noise Mask"
        }
      },
      "478": {
        "inputs": {
          "mask": [
            "481",
            0
          ]
        },
        "class_type": "MaskToImage",
        "_meta": {
          "title": "Convert Mask to Image"
        }
      },
      "479": {
        "inputs": {
          "samples": [
            "510",
            0
          ],
          "vae": [
            "520",
            2
          ]
        },
        "class_type": "VAEDecode",
        "_meta": {
          "title": "VAE Decode"
        }
      },
      "481": {
        "inputs": {
          "expand": -5,
          "incremental_expandrate": 0,
          "tapered_corners": True,
          "flip_input": False,
          "blur_radius": 4,
          "lerp_alpha": 1,
          "decay_factor": 1,
          "fill_holes": False,
          "mask": [
            "558",
            0
          ]
        },
        "class_type": "GrowMaskWithBlur",
        "_meta": {
          "title": "Grow Mask With Blur"
        }
      },
      "482": {
        "inputs": {
          "force_resize_width": 0,
          "force_resize_height": 0,
          "image": [
            "479",
            0
          ],
          "mask": [
            "478",
            0
          ]
        },
        "class_type": "Cut By Mask",
        "_meta": {
          "title": "Cut By Mask"
        }
      },
      "483": {
        "inputs": {
          "text1": "BEFORE",
          "text2": "AFTER",
          "footer_height": 100,
          "font_name": "impact.ttf",
          "font_size": 50,
          "mode": "dark",
          "border_thickness": 20,
          "image1": [
            "548",
            0
          ],
          "image2": [
            "485",
            0
          ]
        },
        "class_type": "CR Simple Image Compare",
        "_meta": {
          "title": "📱 CR Simple Image Compare"
        }
      },
      "485": {
        "inputs": {
          "x": 0,
          "y": 0,
          "resize_source": False,
          "destination": [
            "479",
            0
          ],
          "source": [
            "548",
            0
          ],
          "mask": [
            "481",
            0
          ]
        },
        "class_type": "ImageCompositeMasked",
        "_meta": {
          "title": "ImageCompositeMasked"
        }
      },
      "487": {
        "inputs": {
          "text_input": "",
          "task": "region_caption",
          "fill_mask": True,
          "keep_model_loaded": False,
          "max_new_tokens": 1024,
          "num_beams": 3,
          "do_sample": True,
          "output_mask_select": "",
          "seed": 741411605227530,
          "image": [
            "548",
            0
          ],
          "florence2_model": [
            "494",
            0
          ]
        },
        "class_type": "Florence2Run",
        "_meta": {
          "title": "Florence2Run"
        }
      },
      "494": {
        "inputs": {
          "model": "microsoft/Florence-2-base",
          "precision": "fp16",
          "attention": "sdpa",
          "convert_to_safetensors": False
        },
        "class_type": "DownloadAndLoadFlorence2Model",
        "_meta": {
          "title": "DownloadAndLoadFlorence2Model"
        }
      },
      "497": {
        "inputs": {
          "text": [
            "506",
            0
          ],
          "clip": [
            "509",
            1
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Prompt)"
        }
      },
      "504": {
        "inputs": {
          "text_0": " human face<loc_386><loc_47><loc_609><loc_345>woman<loc_106><loc_0><loc_887><loc_998>  beautiful face, flawless smooth skin, clean natural skin texture, fine pores, subtle tone variation, soft complexion, radiant healthy skin, high detail skin texture, natural glow, photorealistic, masterpiece ",
          "text": [
            "506",
            0
          ]
        },
        "class_type": "ShowText|pysssss",
        "_meta": {
          "title": "Show Text 🐍"
        }
      },
      "506": {
        "inputs": {
          "part1": "",
          "part2": [
            "487",
            2
          ],
          "part3": " beautiful face, flawless smooth skin, clean natural skin texture, fine pores, subtle tone variation, soft complexion, radiant healthy skin, high detail skin texture, natural glow, photorealistic, masterpiece",
          "part4": "",
          "separator": " "
        },
        "class_type": "CR Combine Prompt",
        "_meta": {
          "title": "⚙️ CR Combine Prompt"
        }
      },
      "507": {
        "inputs": {
          "text": "redness, rash, wrinkles, oily skin, pores enlarged, rough texture, plastic skin, waxy skin, airbrushed, fake textures, lowres, blurry, deformed face, face morph, cartoon, cgi, 3d render, doll-like, blur",
          "clip": [
            "509",
            1
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "NEGATIVE"
        }
      },
      "509": {
        "inputs": {
          "PowerLoraLoaderHeaderWidget": {
            "type": "PowerLoraLoaderHeaderWidget"
          },
          "lora_1": {
            "on": False,
            "lora": "epiCRealness.safetensors",
            "strength": 1.04
          },
          "lora_2": {
            "on": False,
            "lora": "epiCRealismXL_KiSS_Enhancer.safetensors",
            "strength": 0.82
          },
          "lora_3": {
            "on": False,
            "lora": "epiCPhotoXL.safetensors",
            "strength": 1.5
          },
          "lora_4": {
            "on": False,
            "lora": "real_humans.safetensors",
            "strength": 1.7
          },
          "➕ Add Lora": "",
          "model": [
            "520",
            0
          ],
          "clip": [
            "520",
            1
          ]
        },
        "class_type": "Power Lora Loader (rgthree)",
        "_meta": {
          "title": "Power Lora Loader (rgthree)"
        }
      },
      "510": {
        "inputs": {
          "seed": 836357257026578,
          "steps": 30,
          "cfg": 2,
          "sampler_name": "dpmpp_2m",
          "scheduler": "karras",
          "denoise": 0.3,
          "model": [
            "509",
            0
          ],
          "positive": [
            "497",
            0
          ],
          "negative": [
            "507",
            0
          ],
          "latent_image": [
            "476",
            0
          ]
        },
        "class_type": "KSampler",
        "_meta": {
          "title": "KSampler"
        }
      },
      "512": {
        "inputs": {
          "amount": 0.4,
          "image": [
            "515",
            0
          ]
        },
        "class_type": "ImageCASharpening+",
        "_meta": {
          "title": "🔧 Image Contrast Adaptive Sharpening"
        }
      },
      "515": {
        "inputs": {
          "intensity": 0.03,
          "scale": 10,
          "temperature": 0,
          "vignette": 0,
          "image": [
            "485",
            0
          ]
        },
        "class_type": "FilmGrain",
        "_meta": {
          "title": "FilmGrain"
        }
      },
      "520": {
        "inputs": {
          "ckpt_name": "epicrealismXL_vxviLastfameDMD2.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {
          "title": "Load Checkpoint"
        }
      },
      "525": {
        "inputs": {
          "filename_prefix": "body1_",
          "images": [
            "512",
            0
          ]
        },
        "class_type": "SaveImage",
        "_meta": {
          "title": "Save Image"
        }
      },
      "530": {
        "inputs": {
          "pixels": [
            "548",
            0
          ],
          "vae": [
            "520",
            2
          ]
        },
        "class_type": "VAEEncode",
        "_meta": {
          "title": "VAE Encode"
        }
      },
      "533": {
        "inputs": {
          "face": True,
          "hair": False,
          "body": True,
          "clothes": False,
          "accessories": False,
          "background": True,
          "confidence": 0.4,
          "detail_method": "VITMatte",
          "detail_erode": 6,
          "detail_dilate": 6,
          "black_point": 0.01,
          "white_point": 0.99,
          "process_detail": True,
          "device": "cuda",
          "max_megapixels": 2,
          "images": [
            "548",
            0
          ]
        },
        "class_type": "LayerMask: PersonMaskUltra V2",
        "_meta": {
          "title": "LayerMask: PersonMaskUltra V2(Advance)"
        }
      },
      "545": {
        "inputs": {
          "image": "body1__00004_.png"
        },
        "class_type": "LoadImage",
        "_meta": {
          "title": "Load Image"
        }
      },
      "548": {
        "inputs": {
          "seed": 2439781946,
          "resolution": 2048,
          "max_resolution": 4096,
          "batch_size": 1,
          "uniform_batch_size": False,
          "color_correction": "lab",
          "temporal_overlap": 0,
          "prepend_frames": 0,
          "input_noise_scale": 0,
          "latent_noise_scale": 0,
          "offload_device": "cuda:0",
          "enable_debug": False,
          "image": [
            "545",
            0
          ],
          "dit": [
            "550",
            0
          ],
          "vae": [
            "549",
            0
          ]
        },
        "class_type": "SeedVR2VideoUpscaler",
        "_meta": {
          "title": "SeedVR2 Video Upscaler (v2.5.24)"
        }
      },
      "549": {
        "inputs": {
          "model": "ema_vae_fp16.safetensors",
          "device": "cuda:0",
          "encode_tiled": False,
          "encode_tile_size": 1024,
          "encode_tile_overlap": 128,
          "decode_tiled": False,
          "decode_tile_size": 1024,
          "decode_tile_overlap": 128,
          "tile_debug": "false",
          "offload_device": "none",
          "cache_model": False
        },
        "class_type": "SeedVR2LoadVAEModel",
        "_meta": {
          "title": "SeedVR2 (Down)Load VAE Model"
        }
      },
      "550": {
        "inputs": {
          "model": "seedvr2_ema_7b_fp16.safetensors",
          "device": "cuda:0",
          "blocks_to_swap": 0,
          "swap_io_components": False,
          "offload_device": "none",
          "cache_model": False,
          "attention_mode": "sdpa"
        },
        "class_type": "SeedVR2LoadDiTModel",
        "_meta": {
          "title": "SeedVR2 (Down)Load DiT Model"
        }
      },
      "558": {
        "inputs": {
          "background": False,
          "skin": False,
          "nose": False,
          "eye_g": False,
          "r_eye": True,
          "l_eye": True,
          "r_brow": False,
          "l_brow": False,
          "r_ear": False,
          "l_ear": False,
          "mouth": True,
          "u_lip": True,
          "l_lip": True,
          "hair": False,
          "hat": False,
          "ear_r": False,
          "neck_l": False,
          "neck": False,
          "cloth": False,
          "result": [
            "560",
            1
          ]
        },
        "class_type": "FaceParsingResultsParser(FaceParsing)",
        "_meta": {
          "title": "FaceParsingResultsParser(FaceParsing)"
        }
      },
      "559": {
        "inputs": {
          "device": "cuda"
        },
        "class_type": "FaceParsingModelLoader(FaceParsing)",
        "_meta": {
          "title": "FaceParsingModelLoader(FaceParsing)"
        }
      },
      "560": {
        "inputs": {
          "model": [
            "559",
            0
          ],
          "processor": [
            "561",
            0
          ],
          "image": [
            "479",
            0
          ]
        },
        "class_type": "FaceParse(FaceParsing)",
        "_meta": {
          "title": "FaceParse(FaceParsing)"
        }
      },
      "561": {
        "inputs": {},
        "class_type": "FaceParsingProcessorLoader(FaceParsing)",
        "_meta": {
          "title": "FaceParsingProcessorLoader(FaceParsing)"
        }
      }
    }
  }
}
