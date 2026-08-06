import os
from PIL import Image

def blend_images(img1, img2, steps):
    blended = []
    for i in range(1, steps + 1):
        alpha = i / (steps + 1.0)
        blended.append(Image.blend(img1, img2, alpha))
    return blended

def generate_gifs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")

    AVATAR_FRAMES = {
        "talking"  : ["avatar_idle.png", "speak_1.png", "speak_2.png", "speak_1.png", "avatar_idle.png"],
        "idle"     : ["avatar_idle.png", "idle_1.png", "avatar_idle.png"],
        "listening": ["listen_1.png", "avatar_idle.png"],
        "thinking" : ["avatar_idle.png", "think_0.png"],
    }

    # Custom durations for the KEY frames
    CUSTOM_DURATIONS = {
        "talking": [100, 100, 100, 100, 100], 
        "idle": [3000, 150, 3000],
        "listening": [400, 2000], 
        "thinking": [1500, 1500],
    }

    # Number of intermediate frames to generate between each keyframe for smoothness
    BLEND_STEPS = 3
    # Duration for each blended frame
    BLEND_DURATION = 30

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
            
        final_images = []
        final_durations = []
        
        durations = CUSTOM_DURATIONS.get(state, [100] * len(key_images))
        
        for i in range(len(key_images)):
            # Add the keyframe
            final_images.append(key_images[i])
            final_durations.append(durations[i])
            
            # If not the last frame, add blended frames transitioning to the next keyframe
            if i < len(key_images) - 1:
                blended = blend_images(key_images[i], key_images[i+1], BLEND_STEPS)
                final_images.extend(blended)
                final_durations.extend([BLEND_DURATION] * BLEND_STEPS)
            elif state == "talking":
                # For talking, loop back to the first frame smoothly
                blended = blend_images(key_images[i], key_images[0], BLEND_STEPS)
                final_images.extend(blended)
                final_durations.extend([BLEND_DURATION] * BLEND_STEPS)
                
        out_path = os.path.join(assets_dir, f"{state}.gif")
        
        final_images[0].save(
            out_path,
            save_all=True,
            append_images=final_images[1:],
            optimize=False,
            duration=final_durations,
            loop=0
        )
        print(f"Generated ultra-smooth {out_path}")

if __name__ == '__main__':
    generate_gifs()
