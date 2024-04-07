import time
from os.path import join, dirname

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn.functional as F
from scipy.ndimage import convolve
from torch import nn
import math
from skimage.io import imshow, imread
from skimage.color import rgb2yuv, rgb2hsv, rgb2gray, yuv2rgb, hsv2rgb
from scipy.signal import convolve2d
import cv2
from collections import Counter

import random

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For CUDA

# Additional steps for PyTorch to ensure determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
RANDOM_SEED = 42
IMG_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_GENERATE_IMAGES = 9
NUM_TIMESTEPS = 1000
MIXED_PRECISION = "fp16"
GRADIENT_ACCUMULATION_STEPS = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        # transforms.Resize((256, 256)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),

        transforms.ToPILImage()
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    new_img = reverse_transforms(image)
    # plt.imshow(new_img)
    return new_img


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \
                                          time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], \
                                        time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, img):
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
    return img


def find_min_norm_vector_key_index(vectors_dict, u):
    min_distance = float('inf')  # Initialize with infinity
    min_key = None
    min_index = None

    for key, vectors in vectors_dict.items():
        for i, v in enumerate(vectors):
            distance = np.linalg.norm(np.array(u) - np.array(v))
            if distance < min_distance:
                min_distance = distance
                min_key = key
                min_index = i

    return min_key, min_index


def getClosestColor(pixel, color_set_rgb):  # Get the closest color for the pixel
    closest_color = None
    cost_init = 10000
    pixel = np.array(pixel)
    for color in color_set_rgb:
        color = np.array(color)
        cost = np.sum((color - pixel) ** 2)
        if cost < cost_init:
            cost_init = cost
            closest_color = color
    return closest_color


def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary='fill',
                           fillvalue=0)
    return image


def convolver_rgb(image, kernel, iterations=1):
    convolved_image_r = multi_convolver(image[:, :, 0], kernel,
                                        iterations)
    convolved_image_g = multi_convolver(image[:, :, 1], kernel,
                                        iterations)
    convolved_image_b = multi_convolver(image[:, :, 2], kernel,
                                        iterations)

    reformed_image = np.dstack((np.rint(abs(convolved_image_r)),
                                np.rint(abs(convolved_image_g)),
                                np.rint(abs(convolved_image_b))))

    return np.array(reformed_image).astype(np.uint8)


sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
gaussian = (1 / 16.0) * np.array([[1., 2., 1.],
                                  [2., 4., 2.],
                                  [1., 2., 1.]])

unique_pixels = {
    (100, 149, 237),
    (105, 105, 105),
    (127, 255, 212),
    (176, 48, 96),
    (176, 224, 230),
    (204, 204, 255),
    (218, 112, 214),
    (255, 0, 0),
    (255, 182, 193),
    (255, 255, 255)
}

colors = {'grey': [(105, 105, 105), (193, 198, 200), (137, 136, 136), (151, 151, 151), (222, 222, 222), (238, 238, 238),
                   (116, 116, 116)],
          'blue': [(100, 149, 237), (122, 145, 186), (149, 161, 178), (137, 154, 173), (162, 178, 190)],
          'purple_light': [(204, 204, 255)],
          'green': [(127, 255, 212), (141, 164, 162), (108, 131, 123), (142, 255, 216), (116, 185, 162),
                    (143, 153, 155), (193, 196, 196), (165, 195, 189), (147, 186, 180)],
          'red': [(255, 0, 0), (153, 122, 127), (142, 99, 141), (167, 87, 106), (174, 131, 137), (145, 116, 114),
                  (179, 130, 141)],
          'purple': [(218, 112, 214)],
          'pink': [(176, 48, 96), (184, 130, 144), (143, 112, 119), (154, 105, 111)],
          'pink_light': [(255, 182, 193)],
          'blue_light': [(176, 224, 230)],
          'white': [(255, 255, 255)]
          }

_model = None


def _get_model():
    global _model
    if _model:
        return _model
    _model = SimpleUnet()
    _model.load_state_dict(torch.load('model_state_dict.pth', map_location='cpu'))
    return _model


def run(image: Image):

    # save the initial image - for performance
    name = int(time.time())
    image.save(f'{name}_0.jpeg')

    # get or load the model
    model = _get_model()

    transformed_image = preprocess(image)
    batch = transformed_image.unsqueeze(0)
    t = torch.Tensor([5]).type(torch.int64)
    image, noise = forward_diffusion_sample(batch, t)
    image = np.array(show_tensor_image(sample_plot_image(model, image)))

    Image.fromarray(image).save(f'{name}_1.jpeg')

    empty_img = np.array(show_tensor_image(transformed_image))

    for i in range(image.shape[0]):
        for j in range(0, image.shape[1] - 1, 2):
            image[i, j + 1] = image[i, j]

    for i in range(0, image.shape[0] - 1, 2):
        for j in range(image.shape[1]):
            image[i + 1, j] = image[i, j]

    image = cv2.medianBlur(image, 3)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_bound1 = (0, 50, 50)
    upper_bound1 = (179, 255, 255)
    mask1 = cv2.inRange(hsv_image, lower_bound1, upper_bound1)

    min_width_height = 1

    # Initialize an empty mask to accumulate all boxes' coverage
    all_boxes_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Find contours from the mask
    for mask in [mask1]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  ## original

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the box meets the minimum size requirements
            if w >= min_width_height and h >= min_width_height:
                # Optionally, find the most common color within this contour
                contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
                pixels = image[contour_mask == 255]
                pixels = [tuple(pixel) for pixel in pixels.reshape(-1, 3)]
                most_common_color, count = Counter(pixels).most_common(1)[0]
                most_common_color = tuple(int(c) for c in most_common_color)

                # Draw a filled rectangle for this contour with its most common color
                cv2.rectangle(image, (x, y), (x + w, y + h), most_common_color, thickness=cv2.FILLED)

                # Update all_boxes_mask to include this box
                cv2.rectangle(all_boxes_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

    # After processing all contours, invert the all_boxes_mask to get non-detected areas
    uncovered_mask = cv2.bitwise_not(all_boxes_mask)

    # Directly apply the uncovered_color to the non-detected areas of img
    uncovered_color = (105, 105, 105)
    image[uncovered_mask == 255] = uncovered_color
    for row in range(image.shape[0]):
        for col in range(image.shape[0]):
            if np.array_equal(np.array(empty_img)[row][col], np.array([255, 255, 255])):
                image[row][col] = np.array([255, 255, 255])
            else:
                key, index = find_min_norm_vector_key_index(colors, image[row][col])
                if key not in ['pink_light']:
                    image[row][col] = np.array(colors[key][0])
                else:
                    image[row][col] = np.array(colors['grey'][0])

    # image_pil = Image.fromarray(image)
    # resized_image = image_pil.resize((256, 256), Image.Resampling.LANCZOS)
    # image = np.array(resized_image)
    #
    # image_pil = Image.fromarray(empty_img)
    # resized_image = image_pil.resize((256, 256), Image.Resampling.LANCZOS)
    # empty_img = np.array(resized_image)

    for row in range(image.shape[0]):
        for col in range(image.shape[0]):
            if np.array_equal(np.array(empty_img)[row][col], np.array([255, 255, 255])):
                image[row][col] = np.array([255, 255, 255])
            if key == 'pink_light':
                image[row][col] = np.array(colors['pink_light'][0])

    image = Image.fromarray(image)
    image.save(f'{name}_2.jpeg')
    return image
