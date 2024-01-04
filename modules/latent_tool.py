import torch
import torchvision
import math
import cv2
import numpy as np

class VAEEncodeForInpaintV2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", ), "mask": ("MASK", ), "mask_blur": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),}}
    RETURN_TYPES = ("LATENT","IMAGE","MASK",)
    FUNCTION = "encode"

    CATEGORY = "OFF"

    def encode(self, vae, pixels, mask, mask_blur=6):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        resized_pixels = pixels.clone()
        
        if mask_blur == 0:
            mask_erosion = mask
            mask_blurred = mask
        else:
            kernel_size = 2 * int(2.5 * mask_blur + 0.5) + 1
            print("off debug ", mask.shape)
            np_mask = mask.numpy()
            np_mask = np_mask.squeeze()
            print("off debug2  " ,np_mask.shape)
            mask_blurred = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), mask_blur)
            mask_blurred = np.expand_dims(mask_blurred,(0,1))
            mask_blurred = torch.from_numpy(mask_blurred)

            kernel_tensor = torch.ones((1, 1, mask_blur, mask_blur))
            padding = math.ceil((mask_blur - 1) / 2)

            #This shrinks the mask area, so that mask_erosion cover smaller area compared to original face mask. it helps better continuity on boundary.
            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)           

            print("OFF Debug : mask erosion - ",mask_erosion.shape)
           
        m = (1.0 - mask.round()).squeeze(1)
            
        #You may think this can be weird, but it is no problem because m is 0 or 1. (discrete)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m 
            pixels[:,:,:,i] += 0.5
                  
        t = vae.encode(pixels)

        return ({"samples":t, "noise_mask": (mask_erosion[:,:,:x,:y].round())}, resized_pixels, mask_blurred, )
