#from https://github.com/sipherxyz/comfyui-art-venture/blob/main/modules/utility_nodes.py

import io
import os
import json
import torch
import base64
import random
import requests
import hashlib
from typing import List, Dict, Tuple

from PIL import Image, ImageOps, ImageFilter
import numpy as np

from .util import pil2tensor

import folder_paths


MAX_RESOLUTION = 8192

def prepare_image_for_preview(image: Image.Image, output_dir: str, prefix=None):
    if prefix is None:
        prefix = "preview_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    # save image to temp folder
    (
        outdir,
        filename,
        counter,
        subfolder,
        _,
    ) = folder_paths.get_save_image_path(prefix, output_dir, image.width, image.height)
    file = f"{filename}_{counter:05}_.png"
    image.save(os.path.join(outdir, file), format="PNG", compress_level=4)

    return {
        "filename": file,
        "subfolder": subfolder,
        "type": "temp",
    }


def cached_image_load_from_url(url: str):
    cache_path = 'template_cache'
    cache_metadata_file = os.path.join(cache_path,'metadata.json')

    if not os.path.exists(cache_path):
      os.mkdir(cache_path)

    cache_metadata = {}

    if os.path.exists(cache_metadata_file):
      with open(cache_metadata_file,"r") as f:
        cache_metadata = json.load(f)

    print(cache_metadata)

    md5 = hashlib.new('md5')
    md5.update(url.encode())

    url_hash = md5.hexdigest()

    header = {}
    if url_hash in cache_metadata:
      etag = cache_metadata[url_hash]
      header['If-None-Match'] = etag

    response = requests.get(url, timeout = 5, headers=header)

    if response.status_code == 304:
      print("yeah cached!")
    elif response.status_code == 200:
      file_path = os.path.join(cache_path,url_hash)
      with open(file_path,"wb") as f:
        f.write(response.content) 
      if 'Etag' not in response.headers:
        cache_metadata[url_hash] = ''
      else:
        cache_metadata[url_hash] = response.headers['ETag']
        print(response.headers['Etag'])
    else:
      raise Exception(response.text)

    with open(cache_metadata_file,"w") as f:
      json.dump(cache_metadata,f)

    return Image.open(os.path.join(cache_path,url_hash))
    

def load_images_from_url(urls: List[str], keep_alpha_channel=False):
    images = []
    masks = []

    for url in urls:
        if url.startswith("data:image/"):
            i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        elif url.startswith("file://"):
            url = url[7:]
            if not os.path.isfile(url):
                raise Exception(f"File {url} does not exist")

            i = Image.open(url)
        elif url.startswith("http://") or url.startswith("https://"):
            i= cached_image_load_from_url(url)
        elif url.startswith("/view?"):
            from urllib.parse import parse_qs

            qs = parse_qs(url[6:])
            filename = qs.get("name", qs.get("filename", None))
            if filename is None:
                raise Exception(f"Invalid url: {url}")

            filename = filename[0]
            subfolder = qs.get("subfolder", None)
            if subfolder is not None:
                filename = os.path.join(subfolder[0], filename)

            dirtype = qs.get("type", ["input"])
            if dirtype[0] == "input":
                url = os.path.join(folder_paths.get_input_directory(), filename)
            elif dirtype[0] == "output":
                url = os.path.join(folder_paths.get_output_directory(), filename)
            elif dirtype[0] == "temp":
                url = os.path.join(folder_paths.get_temp_directory(), filename)
            else:
                raise Exception(f"Invalid url: {url}")

            i = Image.open(url)
        elif url == "":
            continue
        else:
            raise Exception(f"Invalid url: {url}")

        i = ImageOps.exif_transpose(i)
        has_alpha = "A" in i.getbands()
        mask = None

        if "RGB" not in i.mode:
            i = i.convert("RGBA") if has_alpha else i.convert("RGB")

        if has_alpha:
            mask = i.getchannel("A")

            # recreate image to fix weird RGB image
            alpha = i.split()[-1]
            image = Image.new("RGB", i.size, (0, 0, 0))
            image.paste(i, mask=alpha)
            image.putalpha(alpha)

            if not keep_alpha_channel:
                image = image.convert("RGB")
        else:
            image = i

        images.append(image)
        masks.append(mask)

    return (images, masks)


class CachedLoadImageFromUrl:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "TempImageFromUrl"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "keep_alpha_channel": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "output_mode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "list", "label_off": "batch"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    OUTPUT_IS_LIST = (True, True, False)
    RETURN_NAMES = ("images", "masks", "has_image")
    CATEGORY = "OFF"
    FUNCTION = "load_image"

    def load_image(self, url: str, keep_alpha_channel=False, output_mode=False):
        urls = url.strip().split("\n")
        images, masks = load_images_from_url(urls, keep_alpha_channel)
        if len(images) == 0:
            image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            return ([image], [mask], False)

        previews = []
        np_images = []
        np_masks = []

        for image, mask in zip(images, masks):
            # save image to temp folder
            preview = prepare_image_for_preview(image, self.output_dir, self.filename_prefix)
            image = pil2tensor(image)

            if mask:
                mask = np.array(mask).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            previews.append(preview)
            np_images.append(image)
            np_masks.append(mask.unsqueeze(0))

        if output_mode:
            result = (np_images, np_masks, True)
        else:
            has_size_mismatch = False
            if len(np_images) > 1:
                for image in np_images[1:]:
                    if image.shape[1] != np_images[0].shape[1] or image.shape[2] != np_images[0].shape[2]:
                        has_size_mismatch = True
                        break

            if has_size_mismatch:
                raise Exception("To output as batch, images must have the same size. Use list output mode instead.")

            result = ([torch.cat(np_images)], [torch.cat(np_masks)], True)

        return {"ui": {"images": previews}, "result": result}

