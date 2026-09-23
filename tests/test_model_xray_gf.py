import numpy as np
import pytest

from research.model_xray_adapter.gf_representation import weights_to_gf_image

def test_gf_representation_exact_values():
    # 3 floats. Each is 4 bytes. Total 12 bytes.
    # The smallest square >= 12 is 4x4 (16 bytes), meaning 4 padding bytes (zeros).
    
    # 1.0 in float32 is 0x3f800000. In little-endian, bytes are: 00 00 80 3f
    # 2.0 in float32 is 0x40000000. In little-endian, bytes are: 00 00 00 40
    # -1.0 in float32 is 0xbf800000. In little-endian, bytes are: 00 00 80 bf

    weights = np.array([1.0, 2.0, -1.0], dtype=np.float32)
    gf_image = weights_to_gf_image(weights)

    # Check shape
    assert gf_image.shape == (4, 4)

    # Flatten to check byte values
    flat_bytes = gf_image.flatten()
    
    # 1.0
    assert flat_bytes[0] == 0x00
    assert flat_bytes[1] == 0x00
    assert flat_bytes[2] == 0x80
    assert flat_bytes[3] == 0x3f
    
    # 2.0
    assert flat_bytes[4] == 0x00
    assert flat_bytes[5] == 0x00
    assert flat_bytes[6] == 0x00
    assert flat_bytes[7] == 0x40
    
    # -1.0
    assert flat_bytes[8] == 0x00
    assert flat_bytes[9] == 0x00
    assert flat_bytes[10] == 0x80
    assert flat_bytes[11] == 0xbf

    # Padding
    assert flat_bytes[12] == 0x00
    assert flat_bytes[13] == 0x00
    assert flat_bytes[14] == 0x00
    assert flat_bytes[15] == 0x00
