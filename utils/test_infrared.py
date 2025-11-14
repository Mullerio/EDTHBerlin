import numpy as np


def calc(cam_angle, pixels_needed, cam_width):
    lenght_per_pixel = 2.5 / pixels_needed
    actual_width = lenght_per_pixel * cam_width
    return actual_width, actual_width / (2 * np.tan((cam_angle * np.pi / 180) / 2))


import numpy as np

def total_sweep_width(cam_angle, pixels_needed, cam_width, tilt_max_deg=30, step_deg=0.1):
    """
    Calculate total horizontal coverage for a camera sweeping from -tilt_max to +tilt_max.
    
    cam_angle: HFOV in degrees
    pixels_needed: number of pixels the target should occupy
    cam_width: horizontal resolution in pixels
    tilt_max_deg: max tilt angle from nadir (sweep from -tilt_max to +tilt_max)
    step_deg: angular step for summation
    """
    # Base length per pixel at nadir
    length_per_pixel = 2.5 / pixels_needed
    
    angles = np.arange(-tilt_max_deg, tilt_max_deg+step_deg, step_deg)
    widths = length_per_pixel / np.cos(np.deg2rad(angles)) * cam_width
    
    # Approximate total width as sum of incremental widths for each step
    # Here, delta_width = width * (step / cam_angle) to scale contribution of each step
    # We scale each step by ratio of step / HFOV
    incremental_widths = widths * (step_deg / cam_angle)
    
    total_width = np.sum(incremental_widths)
    
    return total_width



if __name__ == "__main__":
    width, hight = calc(60, 4, 348)
    print(f"width: {width}, hight: {hight}")
    # 50° HFOV, 640 px camera, want 50 pixels on target, sweep -30° to +30°
    w = total_sweep_width(60, 4, 640, 45)
    print(f"Total sweep width: {w:.2f} m")



