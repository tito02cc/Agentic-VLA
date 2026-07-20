"""Core module implementation."""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class GraspVisualizer:


    """GraspVisualizer class."""
    def __init__(self, font_size: int = 20):
        self.font_size = font_size
        self._font = None
        
    @property
    def font(self):
        """font function."""
        if self._font is None:
            try:
                # Note

                self._font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", self.font_size)
            except:
                try:
                    self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size)
                except:
                    self._font = ImageFont.load_default()
        return self._font
        
    def draw_grasp_point(
        self,
        image: Image.Image,
        position_2d: Tuple[int, int],
        gripper_width_px: int = 40,
        color: str = "green",
        label: Optional[str] = None,
    ) -> Image.Image:
        """draw_grasp_point function."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        x, y = position_2d
        hw = gripper_width_px // 2
        
        # Note

        draw.ellipse([x-5, y-5, x+5, y+5], fill=color, outline="white")
        
        # Note

        draw.line([x-hw, y-10, x-hw, y+10], fill=color, width=3)
        draw.line([x+hw, y-10, x+hw, y+10], fill=color, width=3)
        draw.line([x-hw, y, x+hw, y], fill=color, width=1)
        
        # Note

        if label:
            draw.text((x+10, y-20), label, fill=color, font=self.font)
            
        return img
        
    def draw_bounding_box(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        color: str = "red",
        label: Optional[str] = None,
        line_width: int = 2,
    ) -> Image.Image:
        """draw_bounding_box function."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        if label:
            # Note

            text_bbox = draw.textbbox((x1, y1-25), label, font=self.font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1-25), label, fill="white", font=self.font)
            
        return img
        
    def visualize_retrieval_results(
        self,
        query_image: Image.Image,
        retrieved_images: List[Image.Image],
        scores: List[float],
        labels: List[str],
        max_cols: int = 4,
        thumbnail_size: Tuple[int, int] = (200, 200),
    ) -> Image.Image:
        """visualize_retrieval_results function."""
        # Note

        query_thumb = query_image.copy()
        query_thumb.thumbnail(thumbnail_size)
        
        retrieved_thumbs = []
        for img in retrieved_images:
            thumb = img.copy()
            thumb.thumbnail(thumbnail_size)
            retrieved_thumbs.append(thumb)
            
        # Note

        n_results = len(retrieved_thumbs)
        n_cols = min(n_results + 1, max_cols)
        n_rows = (n_results + 1 + n_cols - 1) // n_cols
        
        # Note

        padding = 10
        label_height = 30
        cell_w = thumbnail_size[0] + padding * 2
        cell_h = thumbnail_size[1] + padding * 2 + label_height
        
        canvas_w = n_cols * cell_w
        canvas_h = n_rows * cell_h
        
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        
        # Note

        x, y = padding, padding + label_height
        canvas.paste(query_thumb, (x, y))
        draw.text((x, padding), "Query", fill="blue", font=self.font)
        draw.rectangle([x-2, y-2, x+query_thumb.width+2, y+query_thumb.height+2], outline="blue", width=2)
        
        # Note

        for i, (thumb, score, label) in enumerate(zip(retrieved_thumbs, scores, labels)):
            col = (i + 1) % n_cols
            row = (i + 1) // n_cols
            
            x = col * cell_w + padding
            y = row * cell_h + padding + label_height
            
            canvas.paste(thumb, (x, y))
            
            # Note

            text = f"{label[:15]}... ({score:.2f})" if len(label) > 15 else f"{label} ({score:.2f})"
            draw.text((x, row * cell_h + padding), text, fill="black", font=self.font)
            
            # Note

            if score > 0.7:
                color = "green"
            elif score > 0.5:
                color = "orange"
            else:
                color = "gray"
                
            draw.rectangle([x-2, y-2, x+thumb.width+2, y+thumb.height+2], outline=color, width=2)
            
        return canvas
        
    def create_comparison_grid(
        self,
        images: List[Image.Image],
        labels: List[str],
        n_cols: int = 3,
        cell_size: Tuple[int, int] = (300, 300),
    ) -> Image.Image:
        """create_comparison_grid function."""
        n = len(images)
        n_rows = (n + n_cols - 1) // n_cols
        
        padding = 10
        label_height = 30
        cell_w = cell_size[0] + padding * 2
        cell_h = cell_size[1] + padding * 2 + label_height
        
        canvas_w = n_cols * cell_w
        canvas_h = n_rows * cell_h
        
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            col = i % n_cols
            row = i // n_cols
            
            # Note

            thumb = img.copy()
            thumb.thumbnail(cell_size)
            
            x = col * cell_w + padding
            y = row * cell_h + padding + label_height
            
            # Note

            offset_x = (cell_size[0] - thumb.width) // 2
            offset_y = (cell_size[1] - thumb.height) // 2
            
            canvas.paste(thumb, (x + offset_x, y + offset_y))
            draw.text((x, row * cell_h + padding), label, fill="black", font=self.font)
            
        return canvas
        
    def save_visualization(
        self,
        image: Image.Image,
        filepath: str,
        quality: int = 90,
    ) -> None:
        """save_visualization function."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        image.save(filepath, quality=quality)
        print(f"[GraspVisualizer] saved to {filepath}")


def visualize_grasp_prediction(
    scene_image: Image.Image,
    grasp_position_2d: Tuple[int, int],
    confidence: float,
    reasoning: str = "",
    retrieved_images: Optional[List[Image.Image]] = None,
    retrieved_scores: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> Image.Image:
    """visualize_grasp_prediction function."""
    visualizer = GraspVisualizer()
    
    # Note

    color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
    result = visualizer.draw_grasp_point(
        scene_image,
        grasp_position_2d,
        color=color,
        label=f"Conf: {confidence:.2f}",
    )
    
    # Note

    if retrieved_images and retrieved_scores:
        labels = [f"Ref {i+1}" for i in range(len(retrieved_images))]
        retrieval_viz = visualizer.visualize_retrieval_results(
            result,
            retrieved_images,
            retrieved_scores,
            labels,
        )
        result = retrieval_viz
        
    # Note

    if save_path:
        visualizer.save_visualization(result, save_path)
        
    return result
