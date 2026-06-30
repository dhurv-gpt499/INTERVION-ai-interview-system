from PIL import Image
import os

assets_dir = os.path.dirname(__file__)
img_path = os.path.join(assets_dir, "interviewer.png")

if not os.path.exists(img_path):
    print("Error: interviewer.png not found!")
    exit(1)

base_img = Image.open(img_path).convert("RGBA")
w, h = base_img.size

# 1. Idle (Static)
base_img.save(
    os.path.join(assets_dir, "avatar_idle.gif"),
    save_all=True,
    append_images=[base_img],
    duration=1000,
    loop=0
)

# 2. Talking (Fast vertical stretch to simulate jaw movement/talking)
# We will stretch the bottom 60% of the image slightly down.
talk_frame = base_img.copy()
# Crop top 40% (eyes and above)
top_crop = base_img.crop((0, 0, w, int(h * 0.45)))
# Crop bottom 55% (mouth and below)
bottom_crop = base_img.crop((0, int(h * 0.45), w, h))
# Stretch bottom part slightly
stretched_bottom = bottom_crop.resize((w, int(bottom_crop.height * 1.02)), Image.Resampling.LANCZOS)
# Paste back
talk_frame.paste(top_crop, (0, 0))
talk_frame.paste(stretched_bottom, (0, int(h * 0.45)))
# Crop back to original size if it exceeded
talk_frame = talk_frame.crop((0, 0, w, h))

talk_frame.save(
    os.path.join(assets_dir, "avatar_talking.gif"),
    save_all=True,
    append_images=[base_img, talk_frame, base_img, talk_frame],
    duration=150, # Fast flapping
    loop=0
)

# 3. Listening (Slow breathing effect)
# Slightly scale the entire image by 1%
breathe_frame = base_img.resize((int(w * 1.01), int(h * 1.01)), Image.Resampling.LANCZOS)
# Crop center to keep dimensions same
left = (breathe_frame.width - w) // 2
top = (breathe_frame.height - h) // 2
breathe_frame = breathe_frame.crop((left, top, left + w, top + h))

breathe_frame.save(
    os.path.join(assets_dir, "avatar_listening.gif"),
    save_all=True,
    append_images=[base_img, breathe_frame, breathe_frame, base_img],
    duration=600, # Slow breathing
    loop=0
)

print("GIFs generated successfully!")
