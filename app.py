import sys
sys.path.append('.')

from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from utils import iou, sort_and_deduplicate, relation_classes, MLP, show_anns, show_mask
import torch

from ram_train_eval import RamModel,RamPredictor
from mmengine.config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 512
hidden_size = 256
num_classes = 56

# load sam model
sam = build_sam(checkpoint="./checkpoints/sam_vit_h_4b8939.pth").to(device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

# load ram model
model_path = "./checkpoints/epoch12.pth"
config = dict(
    model=dict(
        pretrained_model_name_or_path='bert-base-uncased',
        load_pretrained_weights=False,
        num_transformer_layer=2,
        input_feature_size=256,
        output_feature_size=768,
        cls_feature_size=512,
        num_relation_classes=56,
        pred_type='attention',
        loss_type='multi_label_ce',
    ),
    load_from=model_path,
)
config = Config(config)

class Predictor(RamPredictor):
    def __init__(self,config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model()

    def _build_model(self):
        self.model = RamModel(**self.config.model).to(self.device)
        if self.config.load_from is not None:
            self.model.load_state_dict(torch.load(self.config.load_from, map_location=self.device))
        self.model.train()

model = Predictor(config)


# visualization
def draw_selected_mask(mask, draw):
    color = (255, 0, 0, 153)
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_object_mask(mask, draw):
    color = (0, 0, 255, 153)
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def vis_selected(pil_image, coords):
    # get coords
    coords_x, coords_y = coords.split(',')
    input_point = np.array([[int(coords_x), int(coords_y)]])
    input_label = np.array([1])
    # load image
    image = np.array(pil_image)
    predictor.set_image(image)
    mask1, score1, logit1, feat1 = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    pil_image = pil_image.convert('RGBA')
    mask_image = Image.new('RGBA', pil_image.size, color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    draw_selected_mask(mask1[0], mask_draw)
    pil_image.alpha_composite(mask_image)

    yield [pil_image]


def create_title_image(word1, word2, word3, width, font_path='./assets/OpenSans-Bold.ttf'):
    # Define the colors to use for each word
    color_red = (255, 0, 0)
    color_black = (0, 0, 0)
    color_blue = (0, 0, 255)

    # Define the initial font size and spacing between words
    font_size = 40

    # Create a new image with the specified width and white background
    image = Image.new('RGB', (width, 60), (255, 255, 255))

    # Load the specified font
    font = ImageFont.truetype(font_path, font_size)

    # Keep increasing the font size until all words fit within the desired width
    while True:
        # Create a draw object for the image
        draw = ImageDraw.Draw(image)
        
        word_spacing = font_size / 2
        # Draw each word in the appropriate color
        x_offset = word_spacing
        draw.text((x_offset, 0), word1, color_red, font=font)
        x_offset += font.getsize(word1)[0] + word_spacing
        draw.text((x_offset, 0), word2, color_black, font=font)
        x_offset += font.getsize(word2)[0] + word_spacing
        draw.text((x_offset, 0), word3, color_blue, font=font)
        
        word_sizes = [font.getsize(word) for word in [word1, word2, word3]]
        total_width = sum([size[0] for size in word_sizes]) + word_spacing * 3

        # Stop increasing font size if the image is within the desired width
        if total_width <= width:
            break
            
        # Increase font size and reset the draw object
        font_size -= 1
        image = Image.new('RGB', (width, 50), (255, 255, 255))
        font = ImageFont.truetype(font_path, font_size)
        draw = None

    return image


def concatenate_images_vertical(image1, image2):
    # Get the dimensions of the two images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with the combined height and the maximum width
    new_image = Image.new('RGBA', (max(width1, width2), height1 + height2))

    # Paste the first image at the top of the new image
    new_image.paste(image1, (0, 0))

    # Paste the second image below the first image
    new_image.paste(image2, (0, height1))

    return new_image


def relate_selected(input_image, k, coords):
    # load image
    pil_image = input_image.convert('RGBA')

    w, h = pil_image.size
    if w > 800:
        pil_image.thumbnail((800, 800*h/w))
        input_image.thumbnail((800, 800*h/w))
        coords = str(int(int(coords.split(',')[0]) * 800 / w)) + ',' + str(int(int(coords.split(',')[1]) * 800 / w))
        
    image = np.array(input_image)
    sam_masks = mask_generator.generate(image)
    # get old mask
    coords_x, coords_y = coords.split(',')
    input_point = np.array([[int(coords_x), int(coords_y)]])
    input_label = np.array([1])
    mask1, score1, logit1, feat1 = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    filtered_masks = sort_and_deduplicate(sam_masks)
    filtered_masks = [d for d in sam_masks if iou(d['segmentation'], mask1[0]) < 0.95][:k]
    pil_image_list = []
    
    # run model
    feat = feat1
    for fm in filtered_masks:
        feat2 = torch.Tensor(fm['feat']).unsqueeze(0).unsqueeze(0).to(device)
        feat = torch.cat((feat, feat2), dim=1)
    matrix_output, rel_triplets = model.predict(feat)
    subject_output = matrix_output.permute([0,2,3,1])[:,0,1:]

    for i in range(len(filtered_masks)):
        
        output = subject_output[:,i]
        
        topk_indices = torch.argsort(-output).flatten()
        relation = relation_classes[topk_indices[:1][0]]
        
        mask_image = Image.new('RGBA', pil_image.size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
            
        draw_selected_mask(mask1[0], mask_draw)
        draw_object_mask(filtered_masks[i]['segmentation'], mask_draw)

        current_pil_image = pil_image.copy()
        current_pil_image.alpha_composite(mask_image)
        
        title_image = create_title_image('Red', relation, 'Blue', current_pil_image.size[0])
        concate_pil_image = concatenate_images_vertical(current_pil_image, title_image)
        pil_image_list.append(concate_pil_image)

    yield pil_image_list


def relate_anything(input_image, k):
    # load image
    pil_image = input_image.convert('RGBA')
    w, h = pil_image.size
    if w > 800:
        pil_image.thumbnail((800, 800*h/w))
        input_image.thumbnail((800, 800*h/w))
    image = np.array(input_image)
    sam_masks = mask_generator.generate(image)
    filtered_masks = sort_and_deduplicate(sam_masks)

    feat_list = []
    for fm in filtered_masks:
        feat = torch.Tensor(fm['feat']).unsqueeze(0).unsqueeze(0).to(device)
        feat_list.append(feat)
    feat = torch.cat(feat_list, dim=1).to(device)
    matrix_output, rel_triplets = model.predict(feat)

    pil_image_list = []
    for i, rel in enumerate(rel_triplets[:k]):
        s,o,r = int(rel[0]),int(rel[1]),int(rel[2])
        relation = relation_classes[r]

        mask_image = Image.new('RGBA', pil_image.size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
            
        draw_selected_mask(filtered_masks[s]['segmentation'], mask_draw)
        draw_object_mask(filtered_masks[o]['segmentation'], mask_draw)

        current_pil_image = pil_image.copy()
        current_pil_image.alpha_composite(mask_image)
        
        title_image = create_title_image('Red', relation, 'Blue', current_pil_image.size[0])
        concate_pil_image = concatenate_images_vertical(current_pil_image, title_image)
        pil_image_list.append(concate_pil_image)

    yield pil_image_list

DESCRIPTION = '''# Relate-Anyting

### 🚀 🚀 🚀 This is a demo that combine Meta's Segment-Anything model with the ECCV'22 paper: [Panoptic Scene Graph Generation](https://psgdataset.org/). 

### 🔥🔥🔥 Please star our codebase [openpsg](https://github.com/Jingkang50/OpenPSG) and [RAM](https://github.com/Luodian/RelateAnything) if you find it useful / interesting.
'''

block = gr.Blocks()
block = block.queue()
with block:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil", value="assets/dog.jpg")
            
            with gr.Tab("Relate Anything"):
                num_relation = gr.Slider(label="How many relations do you want to see", minimum=1, maximum=20, value=5, step=1)
                relate_all_button = gr.Button(label="Relate Anything!")

            with gr.Tab("Relate me with Anything"):
                img_input_coords = gr.Textbox(label="Click anything to get input coords")

                def select_handler(evt: gr.SelectData):
                    coords = evt.index
                    return f"{coords[0]},{coords[1]}"

                input_image.select(select_handler, None, img_input_coords)
                run_button_vis = gr.Button(label="Visualize the Select Thing")
                selected_gallery = gr.Gallery(label="Selected Thing", show_label=True, elem_id="gallery").style(preview=True, grid=2, object_fit="scale-down")

                k = gr.Slider(label="Number of things you want to relate", minimum=1, maximum=20, value=5, step=1)
                relate_selected_button = gr.Button(value="Relate it with Anything", interactive=True)

        with gr.Column():
            image_gallery = gr.Gallery(label="Your Result", show_label=True, elem_id="gallery").style(preview=True, columns=5, object_fit="scale-down")

    # relate anything
    relate_all_button.click(fn=relate_anything, inputs=[input_image, num_relation], outputs=[image_gallery], show_progress=True, queue=True)

    # relate selected
    run_button_vis.click(fn=vis_selected, inputs=[input_image, img_input_coords], outputs=[selected_gallery], show_progress=True, queue=True)
    relate_selected_button.click(fn=relate_selected, inputs=[input_image, k, img_input_coords], outputs=[image_gallery], show_progress=True, queue=True)

block.launch(debug=True, share=True)
