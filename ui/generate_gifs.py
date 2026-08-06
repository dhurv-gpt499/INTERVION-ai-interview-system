import os
from PIL import Image

def generate_gifs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")

    AVATAR_FRAMES = {
        "talking"  : ["avatar_idle.png", "speak_1.png", "speak_2.png", "speak_1.png"],
        "idle"     : ["avatar_idle.png", "idle_1.png", "avatar_idle.png"],
        "listening": ["listen_1.png", "avatar_idle.png"],
        "thinking" : ["avatar_idle.png", "think_0.png"],
    }

    # Custom durations for each frame in milliseconds
    CUSTOM_DURATIONS = {
        "talking": [100, 100, 100, 100], 
        "idle": [3000, 150, 3000],
        "listening": [400, 2000], 
        "thinking": [1500, 1500],
    }

    for state, filenames in AVATAR_FRAMES.items():
        key_images = []
        for f in filenames:
            img_path = os.path.join(assets_dir, f)
            if not os.path.exists(img_path):
                print(f"Missing {img_path}")
                continue
            key_images.append(Image.open(img_path).convert("RGBA"))
        
        if len(key_images) < 2:
            continue
            
        durations = CUSTOM_DURATIONS.get(state, [100] * len(key_images))
        out_path = os.path.join(assets_dir, f"{state}.webp")
        
        key_images[0].save(
            out_path,
            save_all=True,
            append_images=key_images[1:],
            duration=durations,
            loop=0
        )
        print(f"Generated crisp animation {out_path}")

if __name__ == '__main__':
    generate_gifs()
