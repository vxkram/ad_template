from fastapi import FastAPI, File, UploadFile, Form, Request
import io, os, uvicorn
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
import base64
import joblib
import pandas as pd
import torch

current_directory = os.path.dirname(os.path.realpath(__file__))
IMAGES_DIR = os.path.join(current_directory, "images")
GENERATED_IMAGE_PATH = os.path.join(IMAGES_DIR, "image_createdx.jpg")
BUTTON_TEMPLATE_PATH = os.path.join(IMAGES_DIR, "buttonx.png")
FONT_PATH = os.path.join(IMAGES_DIR, "arial_narrow_7.ttf")

app = FastAPI()

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
templates = Jinja2Templates(directory=os.path.join(current_directory, "templates"))

color_data = pd.read_csv(os.path.join(current_directory, "ml_color", "color_names.csv"))
color_model = joblib.load(os.path.join(current_directory, "ml_color", "base_model.joblib"))

_pipeline = None
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_pipeline():
    """Loads the image2image pipeline on first use instead of at import time,
    so the app starts instantly and endpoints that don't need it (color-name
    lookup, template compositing) work without downloading multi-GB weights."""
    global _pipeline
    if _pipeline is None:
        from diffusers import AutoPipelineForImage2Image

        device = get_device()
        dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
        _pipeline = AutoPipelineForImage2Image.from_pretrained(
            MODEL_ID, safety_checker=None, torch_dtype=dtype, variant="fp16"
        ).to(device)
    return _pipeline


def hex_to_rgb(hex_string):
    if hex_string[0] == '#':
        hex_string = hex_string[1:]
    r = int(hex_string[0:2], 16)
    g = int(hex_string[2:4], 16)
    b = int(hex_string[4:6], 16)
    return r, g, b


def prompt_with_colorname(prompt, hex):
    r, g, b = hex_to_rgb(hex)
    match = color_data[
        (color_data['Red (8 bit)'] == r) & (color_data['Green (8 bit)'] == g) & (color_data['Blue (8 bit)'] == b)
    ]
    if len(match) > 0:
        predicted_color = match['Name'].values[0]
    else:
        predicted_color = color_model.predict([[r, g, b]])[0]

    return prompt + f', {predicted_color}'


def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, "white")
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im


def create_image(logo, created_path, button_path, button_text, punchline_text, color):
    template_shape = 800
    template = Image.new("RGB", (template_shape, template_shape), "white")

    pading_y = template_shape // 5
    pading_x = int(template_shape / 2)

    logo_shape = template_shape // 9
    logo = logo.resize((logo_shape, logo_shape))
    template.paste(logo, ((pading_x - logo_shape // 2), template_shape // 25), logo)

    created_image = Image.open(created_path)
    created_image_shape = template_shape // 2
    created_image = created_image.resize((created_image_shape, created_image_shape))
    created_image = add_corners(created_image, 30)
    pading_img2 = 2 * template_shape // 30 + logo_shape
    template.paste(created_image, ((pading_x - created_image_shape // 2), pading_img2), created_image)

    # Button
    button = Image.open(button_path).convert("RGB")
    button_shape_x = int(created_image_shape // 1.5)
    button_shape_y = logo_shape // 2
    button = button.resize((button_shape_x, button_shape_y))
    pixel_data = button.getdata()
    recolored_pixels = []
    for pixel in pixel_data:
        if pixel[0] in list(range(0, 50)):
            recolored_pixels.append((color[0], color[1], color[2]))
        else:
            recolored_pixels.append(pixel)

    button.putdata(recolored_pixels)
    button = button.convert('RGBA')
    button_size = template_shape // 80
    button = add_corners(button, button_size)
    template.paste(button, (pading_x - (button_shape_x // 2), (template_shape - (template_shape // 18) - button_shape_y)), button)

    # Button text
    draw = ImageDraw.Draw(template)
    button_text_size = 35 * button_size / len(button_text) * 1.2
    font = ImageFont.truetype(FONT_PATH, button_text_size)
    draw.text(
        (pading_x - (button_shape_x // 2) + (button_shape_x // 4 - button_text_size),
         (template_shape - (template_shape // 18) - button_shape_y + (button_shape_y // 1.2 - button_text_size))),
        button_text, font=font, fill=(255, 255, 255),
    )

    # Punchline text
    draw = ImageDraw.Draw(template)
    font = ImageFont.truetype(FONT_PATH, 50)
    text_length = draw.textlength(text=punchline_text, font=font)
    if text_length > template_shape:
        half_length = len(punchline_text) // 2
        space_index = punchline_text.find(' ', half_length)
        modified_text = punchline_text[:space_index] + '\n' + punchline_text[space_index + 1:]
        half_text_length = draw.textlength(text=punchline_text[:space_index], font=font)
        draw.text((pading_x - half_text_length // 2, (template_shape - (template_shape // 5) - button_shape_y)), modified_text, font=font, fill=(color[0], color[1], color[2]))
    else:
        draw.text((pading_x - text_length // 2, (template_shape - (template_shape // 5) - button_shape_y)), punchline_text, font=font, fill=(color[0], color[1], color[2]))

    # upperline
    line_width, line_height = template_shape // 1.1, 7
    draw = ImageDraw.Draw(template)
    rect_coords = [pading_x - line_width // 2, -line_height, pading_x + line_width // 2, line_height]
    corner_radius = 20
    draw.rounded_rectangle(rect_coords, corner_radius, fill=(color[0], color[1], color[2]))
    # lowerline
    draw = ImageDraw.Draw(template)
    rect_coords = [pading_x - line_width // 2, template_shape - line_height, pading_x + line_width // 2, template_shape + line_height]
    draw.rounded_rectangle(rect_coords, corner_radius, fill=(color[0], color[1], color[2]))
    return template


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "getimg2img.html")


@app.get("/get_output_template", response_class=HTMLResponse)
async def get_output_template_form(request: Request):
    cache_bust = int(os.path.getmtime(GENERATED_IMAGE_PATH)) if os.path.isfile(GENERATED_IMAGE_PATH) else 0
    return templates.TemplateResponse(request, "get_template.html", {"cache_bust": cache_bust})


@app.post("/color_name")
async def color_name(prompt: str = Form("disney pixar"), hex: str = Form("#000000")):
    prompt = prompt_with_colorname(prompt, hex)
    return {"message": f"Welcome to the Text-to-Image API! - {prompt}"}


@app.post("/getimg2img")
async def generate_styled_image(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form("disney pixar"),
    hex: str = Form("#000000"),
):
    try:
        contents = await image.read()
        source_image = Image.open(io.BytesIO(contents))
        if source_image.mode in ('RGBA', 'LA'):
            source_image = source_image.convert("RGB")
        source_image = source_image.resize((512, 512))

        full_prompt = prompt_with_colorname(prompt, hex)
        pipeline = get_pipeline()
        image_created = pipeline(prompt=full_prompt, image=source_image).images[0]

        image_created.save(GENERATED_IMAGE_PATH)

        redirect_url = request.url_for('get_output_template_form')
        return RedirectResponse(url=redirect_url, status_code=303)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/output_template", response_class=HTMLResponse)
async def create_ad_template(
    request: Request,
    logo: UploadFile = File(...),
    hex: str = Form("#000000"),
    punchline_text: str = Form("AI ad banners lead to higher conversions rates"),
    button_text: str = Form("call to action text here"),
):
    try:
        r, g, b = hex_to_rgb(hex)
        contents = await logo.read()
        logo_image = Image.open(io.BytesIO(contents)).convert('RGBA')

        result_image = create_image(
            logo_image,
            GENERATED_IMAGE_PATH,
            BUTTON_TEMPLATE_PATH,
            button_text, punchline_text, [r, g, b],
        )
        buffer = io.BytesIO()
        result_image.save(buffer, 'JPEG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')

        return templates.TemplateResponse(
            request, "result.html", {"image_data": encoded}
        )
    except FileNotFoundError:
        return JSONResponse(
            content={"error": "No styled image yet — generate one on the home page first."},
            status_code=400,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
