"""
Scene condition classification for NuScenes metadata

This module classifies scenes by environmental conditions (day/night/rain)
to enable condition-specific metric aggregation and entropy analysis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class SceneTags:
    """
    Environmental condition tags for a scene
    
    Attributes:
        is_night: Scene occurs at night/dark conditions
        is_rain: Scene has rain/wet conditions
        is_day: Scene occurs during daytime
    """
    is_night: bool
    is_rain: bool
    is_day: bool


def extract_scene_text(scene_rec: dict, log_rec: dict) -> str:
    """
    Extract and normalize all relevant text from scene and log metadata
    
    Args:
        scene_rec: NuScenes scene record (from nusc.get('scene', scene_token))
        log_rec: NuScenes log record (from nusc.get('log', log_token))
    
    Returns:
        Lowercase concatenated text from description, name, and log fields
    """
    text_parts = []
    
    # Scene description (most informative)
    if 'description' in scene_rec and scene_rec['description']:
        text_parts.append(scene_rec['description'])
    
    # Scene name
    if 'name' in scene_rec and scene_rec['name']:
        text_parts.append(scene_rec['name'])
    
    # Log location (might contain weather info)
    if 'location' in log_rec and log_rec['location']:
        text_parts.append(log_rec['location'])
    
    # Log vehicle (unlikely to contain condition info, but check anyway)
    if 'vehicle' in log_rec and log_rec['vehicle']:
        text_parts.append(log_rec['vehicle'])
    
    # Concatenate and normalize
    combined = ' '.join(text_parts).lower()
    
    return combined


def classify_scene(scene_rec: dict, log_rec: dict) -> SceneTags:
    """
    Classify scene environmental conditions from metadata
    
    Classification logic:
    - NIGHT: Keywords like 'night', 'nighttime', 'dark', 'evening', 'dusk'
    - RAIN: Keywords like 'rain', 'rainy', 'drizzle', 'wet', 'precipitation'
    - DAY: Default if not night (default-day policy)
    
    Args:
        scene_rec: NuScenes scene record
        log_rec: NuScenes log record
    
    Returns:
        SceneTags with condition booleans
    """
    text = extract_scene_text(scene_rec, log_rec)
    
    # Detect night conditions
    night_keywords = ['night', 'nighttime', 'dark', 'evening', 'dusk']
    is_night = any(keyword in text for keyword in night_keywords)
    
    # Detect rain conditions
    rain_keywords = ['rain', 'rainy', 'drizzle', 'wet', 'precipitation', 'shower']
    is_rain = any(keyword in text for keyword in rain_keywords)
    
    # Day is complement of night (default-day policy)
    is_day = not is_night
    
    return SceneTags(is_night=is_night, is_rain=is_rain, is_day=is_day)


def groups_for_tags(tags: SceneTags) -> List[str]:
    """
    Generate overlapping condition groups for a scene
    
    Produces the requested buckets:
    - DAY_SCENES or NIGHT_SCENES (mutually exclusive)
    - RAINY_SCENES (if rain detected)
    - RAINY_DAY_SCENES (if rain + day)
    - RAINY_NIGHT_SCENES (if rain + night)
    
    Args:
        tags: Scene condition tags
    
    Returns:
        List of group names this scene belongs to
        
    Example:
        >>> tags = SceneTags(is_night=False, is_rain=True, is_day=True)
        >>> groups_for_tags(tags)
        ['DAY_SCENES', 'RAINY_SCENES', 'RAINY_DAY_SCENES']
    """
    groups = []
    
    # Always assign to either DAY or NIGHT (mutually exclusive)
    if tags.is_day:
        groups.append('DAY_SCENES')
    else:
        groups.append('NIGHT_SCENES')
    
    # Add rain-related groups if rain detected
    if tags.is_rain:
        groups.append('RAINY_SCENES')
        
        # Add combined rain + time-of-day groups
        if tags.is_day:
            groups.append('RAINY_DAY_SCENES')
        else:
            groups.append('RAINY_NIGHT_SCENES')
    
    return groups


def get_scene_groups_from_dataset(dataset, scene_idx: int) -> List[str]:
    """
    Convenience function to get condition groups for a scene from dataset
    
    Args:
        dataset: NuscData dataset (from val_loader.dataset)
        scene_idx: Index into dataset
    
    Returns:
        List of condition group names for this scene
    """
    rec = dataset.ixes[scene_idx]
    scene_rec = dataset.nusc.get('scene', rec['scene_token'])
    log_rec = dataset.nusc.get('log', scene_rec['log_token'])
    
    tags = classify_scene(scene_rec, log_rec)
    return groups_for_tags(tags)
