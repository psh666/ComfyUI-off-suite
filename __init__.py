from .modules.image_loader import CachedLoadImageFromUrl
from .modules.image_tool import OFFCenterCrop, OFFCenterCropSEGS, OFFSEGSToImage, OFFImageResizeFit, OFFWatermark, MaskToImageFallback, MaskDilationForEachFace
from .modules.misc import GWNumFormatter, QueryGenderAge
from .modules.latent_tool import VAEEncodeForInpaintV2


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image Resize Fit":OFFImageResizeFit,
    "Image Crop Fit": OFFCenterCrop,
    "OFF SEGS to Image": OFFSEGSToImage,
    "Crop Center with SEGS" : OFFCenterCropSEGS,
    "Crop Center wigh SEGS": OFFCenterCropSEGS,
    "Watermarking" : OFFWatermark,
    "GW Number Formatting": GWNumFormatter,
    "Cached Image Load From URL": CachedLoadImageFromUrl,
    "VAE Encode For Inpaint V2" : VAEEncodeForInpaintV2, 
    "Query Gender and Age" : QueryGenderAge,
    "Safe Mask to Image" : MaskToImageFallback,
    "Dilate Mask for Each Face": MaskDilationForEachFace, 
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Resize Fit" : "Image Resize Fit",
    "Image Crop Fit": "Image Crop Fit Node",
    "OFF SEGS to Image": "OFF SEGS to Image",
    "Crop Center with SEGS":"Crop Center with SEGS",
    "Crop Center wigh SEGS": "DONT USE THIS",
    "Watermarking" : "Watermarking",
    "GW Number Formatting": "GW Number Formatting Node",
    "Cached Image Load From URL": "Cached Image Load From URL",
    "VAE Encode For Inpaint V2" : "VAE Encode For Inpaint V2",
    "Query Gender and Age" : "Query Gender and Age" ,
    "Safe Mask to Image" : "Safe Mask to Image",
    "Dilate Mask for Each Face": "Dilate Mask for Each Face", 
}
