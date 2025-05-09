# import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import random
import io
import os

# # Convert between PIL and OpenCV
# def pil_to_cv(pil_img):
#     return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# def cv_to_pil(cv_img):
#     return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# 1. Blur
def apply_blur(image, radius=5):
    return image.filter(ImageFilter.GaussianBlur(radius))

# # 2. Lens effect (simple circular blur in center)
# def apply_lens_effect(image):
#     img_cv = pil_to_cv(image)
#     rows, cols, _ = img_cv.shape
#     kernel_x = cv2.getGaussianKernel(cols, 200)
#     kernel_y = cv2.getGaussianKernel(rows, 200)
#     kernel = kernel_y * kernel_x.T
#     mask = 255 * kernel / np.linalg.norm(kernel)
#     blurred = cv2.GaussianBlur(img_cv, (0, 0), sigmaX=15, sigmaY=15)
#     output = img_cv * (mask[..., None] / 255.0) + blurred * (1 - (mask[..., None] / 255.0))
#     return cv_to_pil(np.uint8(output))

# 3. Grayscale
def apply_grayscale(image):
    return ImageOps.grayscale(image).convert("RGB")

# 4. Uncentered crop
def apply_uncentered_crop(image, crop_fraction=0.7):
    w, h = image.size
    crop_w, crop_h = int(w * crop_fraction), int(h * crop_fraction)
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    return image.crop((x, y, x + crop_w, y + crop_h)).resize((w, h))

# 5. Rotation
def apply_rotation(image, angle=45):
    return image.rotate(angle, expand=True).resize(image.size)

# 6. Flip (horizontal or vertical)
def apply_flip(image, direction='horizontal'):
    if direction == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

# 7. Gaussian noise
# def add_gaussian_noise(image, mean=0, std=25):
#     img_cv = pil_to_cv(image)
#     noise = np.random.normal(mean, std, img_cv.shape).astype(np.uint8)
#     noisy = cv2.add(img_cv, noise)
#     return cv_to_pil(noisy)

# 8. Salt and pepper noise
def add_salt_pepper_noise(image, amount=0.02):
    img_np = np.array(image)
    total_pixels = img_np.size // 3
    num_salt = int(amount * total_pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape[:2]]
    img_np[coords[0], coords[1]] = [255, 255, 255]
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape[:2]]
    img_np[coords[0], coords[1]] = [0, 0, 0]
    return Image.fromarray(img_np)

# 9. Color tint
def apply_tint(image, tint_color='red', intensity=0.5):
    r, g, b = image.split()
    if tint_color == 'red':
        r = ImageEnhance.Brightness(r).enhance(1 + intensity)
    elif tint_color == 'blue':
        b = ImageEnhance.Brightness(b).enhance(1 + intensity)
    elif tint_color == 'green':
        g = ImageEnhance.Brightness(g).enhance(1 + intensity)
    return Image.merge("RGB", (r, g, b))

# 10. Overexposure / underexposure
def apply_exposure(image, factor=1.5):  # >1 = overexpose, <1 = underexpose
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# 11. JPEG compression artifacts
def apply_jpeg_artifacts(image, quality=10):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# 12. Occlusion (overlay random black rectangles)
def apply_occlusion(image, num_blocks=3, max_block_size=0.2):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(num_blocks):
        bw = random.randint(10, int(w * max_block_size))
        bh = random.randint(10, int(h * max_block_size))
        x = random.randint(0, w - bw)
        y = random.randint(0, h - bh)
        draw.rectangle([x, y, x + bw, y + bh], fill=(0, 0, 0))
    return img

# 13. Scale variation (resize object to be much smaller in the frame)
def apply_scale_variation(image, scale=0.3):
    w, h = image.size
    new_w, new_h = int(w * scale), int(h * scale)
    object_img = image.resize((new_w, new_h))
    new_img = Image.new("RGB", (w, h), (255, 255, 255))
    x_offset = (w - new_w) // 2
    y_offset = (h - new_h) // 2
    new_img.paste(object_img, (x_offset, y_offset))
    return new_img

base_images_path = "ressources/images_base/"
output_images_path = "ressources/"

# iterate on all images in the base folder
for image_name in os.listdir(base_images_path):
	if image_name.endswith(".jpg"):
		image_path = os.path.join(base_images_path, image_name)
		image = Image.open(image_path)

		# Apply transformations
        
		# cropped around the object (rectangular crop)
		# to do manually

		transformed_image = apply_blur(image, radius=5)
		transformed_image.save(os.path.join(output_images_path, "blurred/" + image_name))

		# transformed_image = apply_lens_effect(image)
		# transformed_image.save(os.path.join(output_images_path, "lens_effect/" + image_name))

		transformed_image = apply_grayscale(image)
		transformed_image.save(os.path.join(output_images_path, "grayscale/" + image_name))

		# transformed_image = apply_uncentered_crop(image, crop_fraction=0.7)
		# transformed_image.save(os.path.join(output_images_path, "uncentered_crop/" + image_name))
		# to do manually instead

		transformed_image = apply_rotation(image, angle=45)
		transformed_image.save(os.path.join(output_images_path, "rotated/" + image_name))

		transformed_image = apply_flip(image, direction='horizontal')
		transformed_image.save(os.path.join(output_images_path, "flipped_horizontal/" + image_name))

		transformed_image = apply_flip(image, direction='vertical')
		transformed_image.save(os.path.join(output_images_path, "flipped_vertical/" + image_name))

		# transformed_image = add_gaussian_noise(image, mean=0, std=25)
		# transformed_image.save(os.path.join(output_images_path, "gaussian_noise/" + image_name))

		transformed_image = add_salt_pepper_noise(image, amount=0.02)
		transformed_image.save(os.path.join(output_images_path, "salt_pepper_noise/" + image_name))

		transformed_image = apply_tint(image, tint_color='red', intensity=0.5)
		transformed_image.save(os.path.join(output_images_path, "tint_red/" + image_name))

		transformed_image = apply_tint(image, tint_color='blue', intensity=0.5)
		transformed_image.save(os.path.join(output_images_path, "tint_blue/" + image_name))

		transformed_image = apply_tint(image, tint_color='green', intensity=0.5)
		transformed_image.save(os.path.join(output_images_path, "tint_green/" + image_name))

		transformed_image = apply_exposure(image, factor=1.5)
		transformed_image.save(os.path.join(output_images_path, "overexposed/" + image_name))

		transformed_image = apply_exposure(image, factor=0.5)
		transformed_image.save(os.path.join(output_images_path, "underexposed/" + image_name))

		transformed_image = apply_jpeg_artifacts(image, quality=10)
		transformed_image.save(os.path.join(output_images_path, "jpeg_artifacts/" + image_name))

		transformed_image = apply_occlusion(image, num_blocks=3, max_block_size=0.2)
		transformed_image.save(os.path.join(output_images_path, "occluded/" + image_name))

		# transformed_image = apply_scale_variation(image, scale=0.3)
		# transformed_image.save(os.path.join(output_images_path, "small_scale/" + image_name))
		# right now it's just a de zoom

		# transformed_image = apply_scale_variation(image, scale=2.5)
		# transformed_image.save(os.path.join(output_images_path, "big_scale/" + image_name))
		# right now it's just a zoom
