import argparse
from PIL import Image
import os
import torch
import cv2
import json
from models.blip2_model import ImageCaptioning
from diffusers import (
    StableDiffusionPipeline,
    T2IAdapter,
    StableDiffusionAdapterPipeline,
    MultiAdapter,
)

import random
import numpy as np
from controlnet_aux import OpenposeDetector


NEGATIVE_PROMPT = 'extra digit, fewer digits, cropped, worst quality, low quality, blurry, pixelated, low resolution, overexposed, underexposed, too dark, too bright, too blurry, too pixelated, too low resolution, too low quality, too high quality, too high resolution, too sharp, too clear, too focused, too contrasty, too saturated'

class BasicCMC:
    def __init__(self, args):
        # Load your big model here
        self.args = args
        self.init_models()
        self.ref_image = None
        self.is_openai_available = False
        self.setup_seed(9999)

    def init_models(self):
        print(self.args)
        if self.args.stable_diffusion_device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16

        print('\033[1;33m' + "Initializing models...".center(50, '-') + '\033[0m')

        # layer1
        if self.args.custom_prompts is None:
            self.image_caption_model = ImageCaptioning(device=self.args.image_caption_device)
        self.ldm = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.data_type,
            custom_pipeline="lpw_stable_diffusion",
        ).to(self.args.stable_diffusion_device)
        self.tokenizer = self.ldm.tokenizer
        self.text_encoder = self.ldm.text_encoder
        self.ldm.safety_checker = lambda images, clip_input: (images, False)

        # layer2
        self.pose_detector = OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators")

        self.pose_adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2iadapter_openpose_sd14v1", 
            varient="fp16",
            torch_dtype=self.data_type
        ).to(self.args.stable_diffusion_device)

        self.color_adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2iadapter_color_sd14v1", 
            varient="fp16",
            torch_dtype=self.data_type
        ).to(self.args.stable_diffusion_device)
        
        self.t2i_pipeline_layer2 = StableDiffusionAdapterPipeline(
            vae=self.ldm.vae,
            text_encoder=self.ldm.text_encoder,
            tokenizer=self.ldm.tokenizer,
            unet=self.ldm.unet,
            adapter=self.pose_adapter,
            scheduler=self.ldm.scheduler,
            safety_checker=self.ldm.safety_checker,
            feature_extractor=self.ldm.feature_extractor
            )        
        self.t2i_pipeline_layer3 = StableDiffusionAdapterPipeline(
            vae=self.ldm.vae,
            text_encoder=self.ldm.text_encoder,
            tokenizer=self.ldm.tokenizer,
            unet=self.ldm.unet,
            adapter=MultiAdapter([self.pose_adapter,self.color_adapter]),
            scheduler=self.ldm.scheduler,
            safety_checker=self.ldm.safety_checker,
            feature_extractor=self.ldm.feature_extractor
            )
        
        print('\033[1;32m' + "Model initialization finished!".center(50, '-') + '\033[0m')

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def image_to_text(self, ref_image):
        image_caption = self.image_caption_model.image_caption(ref_image)
        return image_caption
    
    def pose_detection(self, ref_image):
        pose, pose_text = self.pose_detector(ref_image)
        return pose, pose_text
    
    def write_svg(self, image, filename):
        bmp = Bitmap(np.array(image.convert('1')))
        path = bmp.trace()
        with open(f"{filename}.svg", "w") as fp:
            fp.write(
                f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}" style="background-color:black"> ''')
            parts = []
            for curve in path:
                fs = curve.start_point
                parts.append(f"M{int(fs[0])},{int(fs[1])}")
                for segment in curve.segments:
                    if segment.is_corner:
                        a = segment.c
                        b = segment.end_point
                        parts.append(f"L{int(a[0])},{int(a[1])}L{int(b[0])},{int(b[1])}")
                    else:
                        a = segment.c1
                        b = segment.c2
                        c = segment.end_point
                        parts.append(f"C{int(a[0])},{int(a[1])} {int(b[0])},{int(b[1])} {int(c[0])},{int(c[1])}")
                parts.append("z")
            fp.write(f'<path stroke="none" fill="white" fill-rule="evenodd" d="{"".join(parts)}"/>')
            fp.write("</svg>")
    
    def color_extraction(self,ref_image):
        color = cv2.cvtColor(np.asarray(ref_image), cv2.COLOR_RGB2BGR)
        H = color.shape[0]
        W = color.shape[1]
        color = cv2.resize(color, (W//64, H//64), interpolation=cv2.INTER_CUBIC)
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        color = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        return color

    def text_to_image(self, text):
        generated_image = self.ldm(
            prompt = text,
            num_inference_steps=self.args.num_diffusion_steps,
            negative_prompt = NEGATIVE_PROMPT,
            ).images[0]
        return generated_image
    
    def text_to_image_with_guidance(self, text, image):
        generated_image = self.t2i_pipeline_layer2(
            text,
            image,
            num_inference_steps = self.args.num_diffusion_steps,
            guidance_scale=7.5,
            adapter_conditioning_scale=1,
            negative_prompt=NEGATIVE_PROMPT
            ).images[0]
        return generated_image
    
    def text_to_image_with_multiple_guidance(self, text, image):
        assert len(image) > 1, 'Need more than one guidance image'
        generated_image = self.t2i_pipeline_layer3(
            text,
            image,
            num_inference_steps = self.args.num_diffusion_steps,
            guidance_scale=7.5,
            adapter_conditioning_scale=[1,1],
            negative_prompt=NEGATIVE_PROMPT
            ).images[0]
        return generated_image

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_src','-i', default=None)
    parser.add_argument('--poseq_src','-e',default=None,help='custom edge detection images, won\'t save the edge images if provided')
    parser.add_argument('--text_src','-t', default=None,help='custom captions, won\'t save the text if provided')
    parser.add_argument('--out_image_dir','-o', default='output')

    parser.add_argument('--image_caption_device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--stable_diffusion_device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--num_diffusion_steps',default=50)

    parser.add_argument('--custom_prompts', default=None,help='Specify a json file with custom hierarchical prompts')
    parser.add_argument('--text_semantic_level', default=None,help="Control the semantic level of the hierarchical text, required if custom prompts are passed")

    parser.add_argument('--num_diffusion_steps', default=50)
    parser.add_argument('--disable_layer','-dl', type=int,nargs='+',help='Disable certain layers',default=None)

    parser.add_argument('--no_color_saving','-nc',action='store_true',default=False,help='Do not save the color images')

    args = parser.parse_args()
    processor = BasicCMC(args)
    
    dl_list = [1,1,1]
    if args.disable_layer is not None:
        print('Disabling layers:',args.disable_layer)
        for ls in args.disable_layer:
            dl_list[ls-1] = 0

    if dl_list[0]:
        layer1_outdir = os.path.join(args.out_image_dir, 'layer1')
        if not os.path.exists(layer1_outdir):
            os.makedirs(layer1_outdir)
    if dl_list[1]:
        layer2_outdir = os.path.join(args.out_image_dir, 'layer2')
        if not os.path.exists(layer2_outdir):
            os.makedirs(layer2_outdir)
    if dl_list[2]:
        layer3_outdir = os.path.join(args.out_image_dir, 'layer3')
        if not os.path.exists(layer3_outdir):
            os.makedirs(layer3_outdir)

    if args.poseq_src is None:
        edge_outdir = os.path.join(args.out_image_dir, 'edge')
        if not os.path.exists(edge_outdir):
            os.makedirs(edge_outdir)
    if args.text_src is None and args.custom_prompts is None:
        text_outdir = os.path.join(args.out_image_dir,'text')
        if not os.path.exists(text_outdir):
            os.makedirs(text_outdir)
    if not args.no_color_saving:
        color_outdir = os.path.join(args.out_image_dir,'color')
        if not os.path.exists(color_outdir):
            os.makedirs(color_outdir)

    if args.custom_prompts is not None:
        assert args.text_semantic_level is not None, 'Semantic level not passed'
        # load json
        with open(args.custom_prompts) as f:
            custom_prompts = json.load(f)
            #use a dict to store the prompts: filename, prompt
            custom_prompts_dict = {}
            for item in custom_prompts:
                try:
                    custom_prompts_dict[item['file']] = item["description"][args.text_semantic_level]
                    print('Loaded custom prompts for:',item['file'])
                except:
                    raise ValueError('Semantic level not found in the json file')
    
    pose_faliure = 0
    for image in os.listdir(args.image_src):
        # load image
        ref_img = Image.open(os.path.join(args.image_src, image)).resize((512,512))

        # extract pose
        if args.poseq_src is not None:
            pose_image = Image.open(os.path.join(args.poseq_src,image))
            pose_text = None
        else:
            pose_image,pose_text = processor.pose_detection(ref_img)
            if len(pose_text) == 0:
                print('Pose detection failed for:',image)
                pose_faliure += 1
                continue
            with open(os.path.join(pose_outdir,"text",image.replace('jpg','txt')),'w') as f:
                f.write(json.dumps(pose_text))            
            pose_image.save(os.path.join(pose_outdir,"image",image))
        
        # extract color
        color_image = processor.color_extraction(ref_img)
        if not args.no_color_saving:
            color_image.save(os.path.join(color_outdir,image))

        # extract text
        if args.custom_prompts is not None:
            try: 
                image_caption = custom_prompts_dict[image]
            except: 
                raise ValueError('Prompt not found for:',image, " ,exiting...")
        elif args.text_src is not None:
            try:
                with open(os.path.join(args.text_src,image.replace('jpg','txt'))) as f:
                    image_caption = f.read()
            except:
                raise ValueError('Text not found for:',image, " ,exiting...")
        else:
            image_caption = processor.image_to_text(ref_img)

        # gen layer1
        if dl_list[0]:
            layer1_recon = processor.text_to_image(image_caption)
            layer1_recon.save(os.path.join(layer1_outdir,image))

        # gen layer2
        if dl_list[1]:
            layer2_recon = processor.text_to_image_with_guidance(image_caption, pose_image)
            layer2_recon.save(os.path.join(layer2_outdir,image))

        # gen layer3
        if dl_list[2]:
            layer3_recon = processor.text_to_image_with_multiple_guidance(image_caption, [pose_image, color_image])
            layer3_recon.save(os.path.join(layer3_outdir,image))

        # saving text 
        if args.text_src is None and args.custom_prompts is None:
            with open(os.path.join(text_outdir,image.replace('jpg','txt')),'w') as f:
                f.write(image_caption)

    if pose_faliure > 0:
        print('Pose detection failed for:',pose_faliure,'images')