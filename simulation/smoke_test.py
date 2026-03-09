#!/usr/bin/env python3
"""
Smoke test for radar_simulation module integrity

This script validates that:
1. All modules can be imported
2. Core functions produce expected output shapes and values
3. Ray tracing maintains physical invariants
"""

import sys
import numpy as np


def test_imports():
    """Test that all modules import without errors"""
    print("Testing imports...")
    try:
        import config
        import information_theory
        import ray_tracing
        import radar_sensor
        import beam_selectors
        import visualization
        # radar_simulation imports torch, which is heavier
        print("  ✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_entropy_computation():
    """Test that entropy computation produces expected values"""
    print("Testing entropy computation...")
    from information_theory import compute_entropy
    
    # Test case 1: Certain occupancy (p=1) should have near-zero entropy
    # (small epsilon used in compute_entropy for numerical stability)
    certain = np.ones((10, 10))
    entropy_certain = compute_entropy(certain)
    if not np.allclose(entropy_certain, 0.0, atol=1e-5):
        print(f"  ✗ Entropy of certain state should be ~0, got {entropy_certain.mean()}")
        return False
    
    # Test case 2: Maximum uncertainty (p=0.5) should have entropy = 1 bit
    uncertain = np.ones((10, 10)) * 0.5
    entropy_uncertain = compute_entropy(uncertain)
    if not np.allclose(entropy_uncertain, 1.0, atol=1e-6):
        print(f"  ✗ Entropy of p=0.5 should be ~1, got {entropy_uncertain.mean()}")
        return False
    
    # Test case 3: Symmetric: H(p) = H(1-p)
    p = 0.3
    prob_map = np.ones((10, 10)) * p
    entropy_p = compute_entropy(prob_map)
    entropy_1_minus_p = compute_entropy(1 - prob_map)
    if not np.allclose(entropy_p, entropy_1_minus_p, atol=1e-6):
        print(f"  ✗ Entropy should be symmetric, H({p}) != H({1-p})")
        return False
    
    print("  ✓ Entropy computation validated")
    return True


def test_ray_tracing_invariants():
    """Test that ray tracing maintains physical invariants"""
    print("Testing ray tracing invariants...")
    from ray_tracing import cast_radar_cone
    from config import RadarConfig, SimulationConfig
    
    radar_config = RadarConfig()
    sim_config = SimulationConfig()
    grid_conf = sim_config.grid_conf
    
    # Create a simple ground truth with an obstacle
    H, W = 200, 200
    ground_truth = np.zeros((H, W))
    # Place obstacle at center-front
    ground_truth[120:140, 95:105] = 1.0
    
    # Cast a ray directly forward (azimuth=0)
    result = cast_radar_cone(0.0, ground_truth, grid_conf, radar_config)
    
    # Invariant 1: Result should have all required keys
    required_keys = ['free_space', 'hit', 'shadow', 'hit_range', 'ray_cells', 'azimuth']
    for key in required_keys:
        if key not in result:
            print(f"  ✗ Missing key in ray result: {key}")
            return False
    
    # Invariant 2: Masks should be boolean/binary
    if result['free_space'].dtype != bool:
        print(f"  ✗ free_space should be boolean")
        return False
    if result['hit'].dtype != bool:
        print(f"  ✗ hit should be boolean")
        return False
    if result['shadow'].dtype != bool:
        print(f"  ✗ shadow should be boolean")
        return False
    
    # Invariant 3: free_space, hit, and shadow should be mutually exclusive
    overlap_free_hit = np.any(result['free_space'] & result['hit'])
    overlap_free_shadow = np.any(result['free_space'] & result['shadow'])
    overlap_hit_shadow = np.any(result['hit'] & result['shadow'])
    
    if overlap_free_hit or overlap_free_shadow or overlap_hit_shadow:
        print(f"  ✗ Ray zones should be mutually exclusive")
        return False
    
    # Invariant 4: If there's a hit, hit_range should be not None
    if result['hit'].any() and result['hit_range'] is None:
        print(f"  ✗ hit_range should not be None when hit is detected")
        return False
    
    # Invariant 5: Shape consistency
    if result['free_space'].shape != (H, W):
        print(f"  ✗ Mask shape mismatch")
        return False
    
    print("  ✓ Ray tracing invariants validated")
    return True


def test_bayesian_fusion():
    """Test that Bayesian fusion produces valid probabilities"""
    print("Testing Bayesian fusion...")
    from information_theory import bayesian_fusion_raytracing
    from ray_tracing import radar_inverse_sensor_model
    from config import RadarConfig, SimulationConfig
    
    radar_config = RadarConfig()
    sim_config = SimulationConfig()
    grid_conf = sim_config.grid_conf
    
    # Create simple prior and ground truth
    H, W = 200, 200
    prior = np.ones((H, W)) * 0.5  # Uniform uncertainty
    ground_truth = np.zeros((H, W))
    ground_truth[120:140, 95:105] = 1.0
    
    # Simulate a radar measurement
    measurement = radar_inverse_sensor_model(
        0.0, ground_truth, grid_conf, radar_config
    )
    
    # Fuse
    posterior = bayesian_fusion_raytracing(prior, measurement, confidence=0.85)
    
    # Invariant 1: Output should be valid probabilities [0, 1]
    if not np.all((posterior >= 0) & (posterior <= 1)):
        print(f"  ✗ Posterior contains invalid probabilities")
        return False
    
    # Invariant 2: Shape should match input
    if posterior.shape != prior.shape:
        print(f"  ✗ Posterior shape mismatch")
        return False
    
    # Invariant 3: Free space should decrease belief
    free_space_prior = prior[measurement['free_space']].mean()
    free_space_posterior = posterior[measurement['free_space']].mean()
    if free_space_posterior >= free_space_prior:
        print(f"  ✗ Free space should decrease occupancy belief")
        return False
    
    print("  ✓ Bayesian fusion validated")
    return True


def run_all_tests():
    """Run all smoke tests"""
    print("\n" + "="*60)
    print("RADAR SIMULATION SMOKE TEST")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_entropy_computation,
        test_ray_tracing_invariants,
        test_bayesian_fusion,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append(False)
        print()
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(results)
    
    for i, (test, passed) in enumerate(zip(tests, results), 1):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test.__name__}")
    
    print("="*60 + "\n")
    
    if all_passed:
        print("🎉 All smoke tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
