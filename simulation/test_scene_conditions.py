"""
Quick validation script for scene condition classification

Tests the scene_conditions module functionality without running full simulation.
"""

from scene_conditions import SceneTags, classify_scene, groups_for_tags

def test_classification_logic():
    """Test scene classification logic with mock data"""
    print("Testing scene classification logic...")
    
    # Test 1: Day scene (no night/rain keywords)
    scene_rec = {'description': 'sunny morning drive', 'name': 'scene-0001'}
    log_rec = {'location': 'boston', 'vehicle': 'v1'}
    tags = classify_scene(scene_rec, log_rec)
    groups = groups_for_tags(tags)
    
    assert tags.is_day == True
    assert tags.is_night == False
    assert tags.is_rain == False
    assert groups == ['DAY_SCENES']
    print("✓ Test 1: Day scene - PASS")
    
    # Test 2: Night scene
    scene_rec = {'description': 'night time city', 'name': 'scene-0002'}
    log_rec = {'location': 'boston', 'vehicle': 'v1'}
    tags = classify_scene(scene_rec, log_rec)
    groups = groups_for_tags(tags)
    
    assert tags.is_day == False
    assert tags.is_night == True
    assert tags.is_rain == False
    assert groups == ['NIGHT_SCENES']
    print("✓ Test 2: Night scene - PASS")
    
    # Test 3: Rainy day scene
    scene_rec = {'description': 'rainy afternoon', 'name': 'scene-0003'}
    log_rec = {'location': 'boston', 'vehicle': 'v1'}
    tags = classify_scene(scene_rec, log_rec)
    groups = groups_for_tags(tags)
    
    assert tags.is_day == True
    assert tags.is_night == False
    assert tags.is_rain == True
    assert set(groups) == {'DAY_SCENES', 'RAINY_SCENES', 'RAINY_DAY_SCENES'}
    print("✓ Test 3: Rainy day scene - PASS")
    
    # Test 4: Rainy night scene
    scene_rec = {'description': 'nighttime drizzle', 'name': 'scene-0004'}
    log_rec = {'location': 'singapore', 'vehicle': 'v1'}
    tags = classify_scene(scene_rec, log_rec)
    groups = groups_for_tags(tags)
    
    assert tags.is_day == False
    assert tags.is_night == True
    assert tags.is_rain == True
    assert set(groups) == {'NIGHT_SCENES', 'RAINY_SCENES', 'RAINY_NIGHT_SCENES'}
    print("✓ Test 4: Rainy night scene - PASS")
    
    # Test 5: Default day (no keywords)
    scene_rec = {'description': '', 'name': 'scene-0005'}
    log_rec = {'location': '', 'vehicle': ''}
    tags = classify_scene(scene_rec, log_rec)
    groups = groups_for_tags(tags)
    
    assert tags.is_day == True  # Default to day
    assert tags.is_night == False
    assert tags.is_rain == False
    assert groups == ['DAY_SCENES']
    print("✓ Test 5: Default day (no keywords) - PASS")
    
    print("\n✅ All scene classification tests passed!")


def test_helper_functions():
    """Test helper function imports"""
    print("\nTesting helper function imports...")
    
    # Test that radar_simulation helpers exist
    from radar_simulation import _create_empty_results_dict, _append_scene_to_condition_buckets
    
    # Create empty results
    results = _create_empty_results_dict()
    assert 'f1_score' in results
    assert 'by_condition' not in results  # by_condition is added separately
    assert isinstance(results['f1_score'], list)
    assert len(results['f1_score']) == 0
    print("✓ _create_empty_results_dict works")
    
    # Test append logic
    results['f1_score'].append(0.85)
    results['iou'].append(0.75)
    results_by_condition = {}
    
    _append_scene_to_condition_buckets(results, results_by_condition, ['DAY_SCENES', 'RAINY_SCENES'])
    
    assert 'DAY_SCENES' in results_by_condition
    assert 'RAINY_SCENES' in results_by_condition
    assert results_by_condition['DAY_SCENES']['f1_score'][-1] == 0.85
    assert results_by_condition['RAINY_SCENES']['iou'][-1] == 0.75
    print("✓ _append_scene_to_condition_buckets works")
    
    print("\n✅ All helper function tests passed!")


if __name__ == '__main__':
    test_classification_logic()
    test_helper_functions()
    print("\n" + "="*60)
    print("🎉 ALL VALIDATION TESTS PASSED!")
    print("="*60)
