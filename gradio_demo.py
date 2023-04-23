import matplotlib.pyplot
import pycocotools
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from utils import iou, sort_and_deduplicate, relation_classes, MLP, show_anns, show_mask
import torch
import matplotlib.pyplot as plt
import random

input_size = 512
hidden_size = 256
num_classes = 56

sam = build_sam(checkpoint="segment_anything/sam_vit_h_4b8939.pth").to('cuda')
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

model = MLP(input_size, hidden_size, num_classes).to('cuda')
model.load_state_dict(torch.load("ram/best_model.pth"))

def make_gif(image, fps=5):
    pass

def draw_mask(mask, draw):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.show()
def inference(pil_image, coords, k=3):
    width, height = pil_image.size
    full_area = width * height
    coords_x, coords_y = coords.split(',')
    input_point = np.array([[int(coords_x), int(coords_y)]])
    input_label = np.array([1])
    image = np.array(pil_image)
    predictor.set_image(image)
    mask1, score1, logit1, feat1 = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    sam_masks = mask_generator.generate(image)
    # remove very small items, duplicate items, and itself
    filtered_masks = [d for d in sam_masks if iou(d['segmentation'], mask1[0]) < 0.95]
    filtered_masks = [d for d in filtered_masks if d['area'] > 0.03 * full_area]
    filtered_masks = sort_and_deduplicate(filtered_masks)
    print('number of masks left:', len(filtered_masks))

    # compute relations

    pil_image = pil_image.convert('RGBA')
    pil_image_list = []
    relation_list = []
    for i in range(len(filtered_masks)):
        mask_image = Image.new('RGBA', pil_image.size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        draw_mask(mask1[0], mask_draw)
        draw_mask(filtered_masks[i]['segmentation'], mask_draw)

        feat2 = torch.Tensor(filtered_masks[i]['feat']).unsqueeze(0).unsqueeze(0).to('cuda')
        concat_input = torch.cat((feat1, feat2), dim=2)
        output = model(concat_input)
        topk_indices = torch.argsort(-output).flatten()
        strings = [relation_classes[indice] for indice in topk_indices[:k]]
        result = ', '.join(strings)
    
        current_pil_image = pil_image.copy()
        # text_draw = ImageDraw.Draw(current_pil_image)
        # text_draw.text((width // 2, height - 20), result, align="center", font=font)
        current_pil_image.alpha_composite(mask_image)
        pil_image_list.append(current_pil_image)
        relation_list.append(result)
    
    # return [pil_image_list, relation_list]
    yield pil_image_list, relation_list


block = gr.Blocks()
block = block.queue()
with block:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil", value="images/dog.jpg")
            img_input_coords = gr.Textbox(label="Input coords")
            k = gr.Slider(label="TopK Relations", minimum=1, maximum=10, value=3, step=1)
            # clear_button = gr.Button(value="Clear Click", interactive=True)

            def select_handler(evt: gr.SelectData):
                coords = evt.index
                # draw = ImageDraw.Draw(image)
                # draw.ellipse((coords[0] - 15, coords[1] - 15, coords[0] + 15, coords[1] + 15), fill="red")
                return f"{coords[0]},{coords[1]}"

            input_image.select(select_handler, None, img_input_coords)

            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)
                text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001)
                iou_threshold = gr.Slider(label="IOU Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.001)
                inpaint_mode = gr.Dropdown(["merge", "first"], value="merge", label="inpaint_mode")

        with gr.Column():
            image_gallery = gr.Gallery(label="Generated images", show_label=True, elem_id="gallery").style(preview=True, grid=2, object_fit="scale-down")
            text_gallery = gr.outputs.Textbox(label="Generated relations")

    run_button.click(fn=inference, inputs=[input_image, img_input_coords, k], outputs=[image_gallery, text_gallery], show_progress=True, queue=True)
    # clear_button.click(fn=reset_image, inputs=[input_image], outputs=input_image)

block.launch(debug=True, share=True)
