
import torch
import math
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from .util import tensor2pil, pil2tensor, empty_pil_tensor, crop_image
import cv2

class OFFImageResizeFit:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",)
            },
            "optional":{
                "size":("INT",{
                    "default": 512,
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "doit"
    CATEGORY = "OFF"

    def doit(self, image, size=512):
        image = tensor2pil(image)
        new_width = size
        new_height = size
        ratio = float(image.width) / float(image.height)
        
        if image.width > image.height :
            new_width = int(size* ratio)
        else:
            new_height = int(size/ratio)

    
        image = image.resize((new_width, new_height))

        return (pil2tensor(image),)



class OFFCenterCrop:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "image_crop"

    # OUTPUT_NODE = False

    CATEGORY = "OFF"

    # Tensor to PIL

    def image_crop(self, image):
        image = tensor2pil(image)
        img_width, img_height = image.size

        crop_size = img_width
        if img_width > img_height:
            crop_size = img_height

        top = (img_height - crop_size)/2
        left = (img_width - crop_size)/2
        bottom = top + crop_size
        right = left + crop_size

        # Calculate the final coordinates for cropping
        crop_top = max(top, 0)
        crop_left = max(left, 0)
        crop_bottom = min(bottom, img_height)
        crop_right = min(right, img_width)

        # Ensure that the cropping region has non-zero width and height
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError(
                "Invalid crop dimensions. Please check the values for top, left, right, and bottom.")

        # Crop the image and resize
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))

        crop = crop.resize(
            (((crop.size[0] // 8) * 8), ((crop.size[1] // 8) * 8)))

        return (pil2tensor(crop),)


class OFFSEGSToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "segs": ("SEGS", ),
        },
            "optional": {
            "fallback_image_opt": ("IMAGE", ),
        }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "OFF"

    def doit(self, segs, fallback_image_opt=None):
        results = list()

        for seg in segs[1]:
            if seg.cropped_image is not None:
                cropped_image = torch.from_numpy(seg.cropped_image)
            elif fallback_image_opt is not None:
                # take from original image
                cropped_image = torch.from_numpy(
                    crop_image(fallback_image_opt, seg.crop_region))
            else:
                cropped_image = empty_pil_tensor()

            results.append(cropped_image)

        if len(results) == 0:
            results.append(empty_pil_tensor())

        return (results[0],)
    
class OFFCenterCropSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "segs": ("SEGS", ),
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "OFF"

    def doit(self, segs, image=None):
        results = list()
        image = tensor2pil(image)
        img_width, img_height = image.size

        crop_size = img_width
        if img_width > img_height:
            crop_size = img_height
            
        for seg in segs[1]:
            
            if seg.cropped_image is not None:
                center_x = (seg.crop_region[0] + seg.crop_region[2])/2
                center_y = (seg.crop_region[1] + seg.crop_region[3])/2

                crop_top = center_y  - crop_size/2
                crop_left = center_x - crop_size/2
                crop_bottom = center_y + crop_size/2
                crop_right = center_x + crop_size/2

                if crop_top < 0:
                    crop_bottom = crop_bottom - crop_top
                    crop_top = 0
                if crop_left < 0:
                    crop_right = crop_right - crop_left
                    crop_left = 0

                if crop_bottom > img_height:
                    crop_top = crop_top - (crop_bottom - img_height)
                    crop_bottom = img_height
                if crop_right > img_width:
                    crop_left = crop_left - (crop_right - img_width)
                    crop_right = img_width

                cropped_image = image.crop(
                    (crop_left, crop_top, crop_right, crop_bottom))
                cropped_image = cropped_image.resize(
                    (((cropped_image.size[0] // 8) * 8), ((cropped_image.size[1] // 8) * 8)))
                cropped_image = pil2tensor(cropped_image)
            else:
                cropped_image = empty_pil_tensor()

            results.append(cropped_image)

        if len(results) == 0:
            results.append(empty_pil_tensor())

        return (results[0],)

class OFFWatermark:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "source":("IMAGE",),
                "watermark":("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "OFF"

    def doit(self, source, watermark):
        source_image = tensor2pil(source)
        watermark_image = tensor2pil(watermark).convert()

        watermark_ratio = watermark_image.height/ watermark_image.width
        new_wm_with = source_image.width*0.15
        watermark_image = watermark_image.resize((int(new_wm_with), int(new_wm_with*watermark_ratio)))
        source_image.paste(watermark_image,(10,10), watermark_image)
            
        return (pil2tensor(source_image),)

class MaskToImageFallback:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "image": ("IMAGE",),
                }
        }

    CATEGORY = "OFF"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_to_image"

    def mask_to_image(self, mask, image):
        if mask is not None:
            result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        else:
            zero_tensor = torch.zeros_like(image)
            result = zero_tensor
        return (result,)
    


def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return mask

    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)
    mask = cv2.UMat(mask)
    kernel = cv2.UMat(kernel)

    result = cv2.erode(mask, kernel, iter)

    return result.get()
    

class MaskDilationForEachFace:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required": {
                    "mask": ("MASK",),
                    "segs": ("SEGS",),
                    "dilation_factor": ("FLOAT",{"min": 0.0, "max": 10.0, "step": 0.1, "default": 1.0})
                }
        }
    CATEGORY = "OFF"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"

    def process(self, mask, segs, dilation_factor):
        if mask.shape[0] is not len(segs[1]):
            return (mask,) 
        
        mask = mask.clone()
        for i in range(0, mask.shape[0]):
            
            crop_mask = torch.zeros_like(mask[i])

            crop_left = segs[1][i].crop_region[0]
            crop_right = segs[1][i].crop_region[2]
            crop_top = segs[1][i].crop_region[1]
            crop_bottom = segs[1][i].crop_region[3]

            crop_mask[:,segs[1][i].crop_region[1]:segs[1][i].crop_region[3] , segs[1][i].crop_region[0]:segs[1][i].crop_region[2]] = 255
            pixel_count = (crop_right - crop_left) * (crop_bottom - crop_top)
            dilation = int(math.sqrt(pixel_count)/3.0/25.0 * dilation_factor)
            print(f"face {i} : pixel_count , dilation =  ", pixel_count, dilation)
            mask[i] = mask[i] * crop_mask

            cv_mask = mask[i].squeeze(0).numpy()
            cv_mask = dilate_mask(cv_mask, dilation_factor=dilation)
            mask[i] = torch.from_numpy(cv_mask).unsqueeze(0)

        return (mask,)

class SegsToFaceCropData:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required":{
                "segs":("SEGS",),
                "face_index":("INT", {"min":0, "default":0}),
            }
        }
    CATEGORY = "OFF"
    RETURN_TYPES = ("IMAGE","CROP_DATA",)
    FUNCTION ="process"

    def process(self, segs, face_index):
        return (segs[1][face_index].cropped_image, segs[1][face_index].crop_region,)

class PasteFaceSegToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "orig_image":("IMAGE",),
                "crop_image":("IMAGE",),
                "crop_data":("CROP_DATA",),
            }
        }
    
    CATEGORY = "OFF"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    def process(self, orig_image, crop_image, crop_data):
        orig_image[:,crop_data[1]: crop_data[1]+crop_image.shape[1], crop_data[0]:crop_data[0]+crop_image.shape[2],:] = crop_image
        return (orig_image,)
    
class OFFCLAHE:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required":{
                "image":("IMAGE",),
                "clip_limit":("FLOAT",{"default":1.2}),
                "grid_size":("INT",{"min":1, "default":8})
            }
        }
    
    CATEGORY =  "OFF"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    def process(self, image, clip_limit, grid_size):
        
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for i in range(batch_size):
            tensor_image = image[i].numpy()
            source_image = np.array(Image.fromarray((tensor_image*255).astype(np.uint8)))


            lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)       
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            result[i] = torch.from_numpy(enhanced_image.astype(np.float32) / 255.0)
        return (result,)
        
class OffGridImageBatch:
    #from WAS SUITE
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "border_width": ("INT", {"default":20, "min": 0, "max": 100, "step":1}),
                "number_of_columns": ("INT", {"default":3, "min": 1, "max": 24, "step":1}),
                "max_cell_size": ("INT", {"default":320, "min":32, "max":2048, "step":1}),
                "border_red": ("INT", {"default":38, "min": 0, "max": 255, "step":1}),
                "border_green": ("INT", {"default":38, "min": 0, "max": 255, "step":1}),
                "border_blue": ("INT", {"default":38, "min": 0, "max": 255, "step":1}),
                "border_radius": ("INT", {"default":24, "min":0, "max":128,"step":1}), 
                "target_size": ("INT", {"default":1080,"min":0,"max":8192,"step":8})
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "smart_grid_image"
    
    CATEGORY = "OFF"

    def smart_grid_image(self, images, number_of_columns=3, max_cell_size=320, border_red=38, border_green=38, border_blue=38, border_width=20, border_radius=24, target_size=1080):
        
        cols = number_of_columns
        border_color = (border_red, border_green, border_blue)

        images_resized = []
        max_row_height = 0
        
        for tensor_img in images:
            img = tensor2pil(tensor_img)
            img_w, img_h = img.size
            aspect_ratio = img_w / img_h
            
            if img_w > img_h:
                cell_w = min(img_w, max_cell_size)
                cell_h = int(cell_w / aspect_ratio)
            else:
                cell_h = min(img_h, max_cell_size)
                cell_w = int(cell_h * aspect_ratio)
            
            img_resized = img.resize((cell_w, cell_h))

            mask = Image.new("L", img_resized.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle((0,0,cell_w, cell_h),fill=255,radius=border_radius)
            back_color = Image.new(img_resized.mode, img_resized.size, border_color)
            img_resized = Image.composite(img_resized, back_color, mask)
            
            img_resized = ImageOps.expand(img_resized, border=border_width // 2, fill=border_color)


            images_resized.append(img_resized)
            max_row_height = max(max_row_height, cell_h)
            
        max_row_height = int(max_row_height)
        total_images = len(images_resized)
        rows = math.ceil(total_images / cols)

        grid_width = cols * max_cell_size + (cols - 1) * border_width
        grid_height = rows * max_row_height + (rows - 1) * border_width
        
        new_image = Image.new('RGB', (grid_width, grid_height), border_color)
        rand_idx = np.arange(len(images_resized))
        np.random.shuffle(rand_idx)
        
        for i, img in enumerate(images_resized):
            x = (rand_idx[i] % cols) * (max_cell_size + border_width)
            y = (rand_idx[i] // cols) * (max_row_height + border_width)
            
            img_w, img_h = img.size
            paste_x = x + (max_cell_size - img_w) // 2
            paste_y = y + (max_row_height - img_h) // 2

            new_image.paste(img, (paste_x, paste_y, paste_x + img_w, paste_y + img_h))

        new_w, _ = new_image.size
        outer_border_width = int((target_size - new_w)/2)

        if outer_border_width>0 :
            new_image = ImageOps.expand(new_image, border=outer_border_width, fill=border_color)

        return (pil2tensor(new_image), )
    
class CalcMaskBound:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "mask": ("MASK",),
                    "padding":("INT",{"default":0, "min": 0, "step":1}),
                    "ratio" : ("FLOAT",{"default":1.0, "min":0.7, "max":1.5, "step":0.1})
                }
        }
    
    CATEGORY = "OFF"

    RETURN_TYPES = ("INT","INT","INT","INT","INT","INT",)
    FUNCTION = "process"

    def process(self, mask, padding, ratio):
        
        rows, cols = np.where(mask[0])
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        min_col -= padding
        min_row -= padding

        max_col += padding
        max_row += padding
        
        if min_col< 0 :
            min_col = 0
        
        if min_row<0:
            min_row = 0
        
        if max_row > mask.shape[1]:
            max_row = mask.shape[1]
            
        if max_col > mask.shape[2]:
            max_col = mask.shape[2]

        ori_width = max_col - min_col
        ori_height = max_row - min_row

        desired_height = max(ori_width, ori_height)
       
        desired_width = int(desired_height * ratio)

        shrink_ratio_w = ori_width/desired_width
        shrink_ratio_h = ori_height/desired_height

        shrink_ratio = max(shrink_ratio_w, shrink_ratio_h)

        desired_width *= shrink_ratio
        desired_height *= shrink_ratio
        
        optional_padding_col = int((desired_width - ori_width) / 2)
        optional_padding_row = int((desired_height - ori_height)/2)

        min_col -= optional_padding_col
        min_row -= optional_padding_row

        max_col += optional_padding_col
        max_row += optional_padding_row

        if min_col < 0 :
            min_col = 0
            max_col -= min_col
        
        if min_row<0:
            min_row = 0
            max_row -= min_row
        
        if max_row > mask.shape[1]:
            min_row -= (mask.shape[1] - max_row)
            max_row = mask.shape[1]
            
            
        if max_col > mask.shape[2]:
            min_col -= (mask.shape[1] - max_col)
            max_col = mask.shape[2]
           
        return (min_col, min_row, max_col - min_col, max_row - min_row, max_col, max_row,)


                





