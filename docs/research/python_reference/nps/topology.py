"""
Neuron Permutation Steganography (NPS) - Topology & Core
Independent reimplementation of function-invariant permutations.

Algorithm:
  W' = P * W
  b' = P * b
  W_next' = W_next * P^T
  
For ResNets, skip connections require permutations to be synchronized
across residual additions.
"""

import numpy as np
import math

def generate_permutation_matrix(n: int, seed: int = None) -> np.ndarray:
    """Generate a random NxN permutation matrix."""
    rng = np.random.default_rng(seed)
    p = np.eye(n)
    rng.shuffle(p)
    return p

def encode_bits_to_permutation(bits: np.ndarray, n: int) -> np.ndarray:
    """
    Encode a bit sequence into a permutation of size n using factoradic (Lehmer) coding.
    """
    if len(bits) == 0:
        return np.eye(n)
        
    # Convert bits to a single integer payload
    payload = 0
    for i, bit in enumerate(bits):
        payload |= (int(bit) << i)
        
    # Compute max capacity
    # For n=512, log2(n!) is ~3600
    # Make sure we don't exceed it
    if payload >= math.factorial(n):
        raise ValueError("Payload exceeds permutation capacity")
        
    # Convert integer to factoradic (Lehmer code)
    lehmer = []
    rem = payload
    for i in range(1, n + 1):
        lehmer.append(rem % i)
        rem //= i
    lehmer.reverse() # This gives us the Lehmer code sequence of length n
    
    # Convert Lehmer code to permutation
    # Lehmer code element i tells us the number of available elements 
    # strictly greater than the element we should pick at index i.
    # Standard Lehmer: l_i is the number of elements in the permutation after index i that are smaller than P[i].
    # Which translates to: pick the l_i-th element from the available ordered choices.
    choices = list(range(n))
    perm = []
    for l_idx in lehmer:
        # l_idx is from 0 to current available choices length - 1
        val = choices.pop(l_idx)
        perm.append(val)
        
    # Convert permutation array to matrix
    p_mat = np.zeros((n, n), dtype=np.float32)
    for i, p_val in enumerate(perm):
        p_mat[i, p_val] = 1.0
        
    return p_mat

def decode_permutation_to_bits(P: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Decode a permutation matrix back into a bit sequence of length num_bits.
    """
    n = P.shape[0]
    
    # Extract permutation array
    perm = []
    for i in range(n):
        perm.append(int(np.argmax(P[i])))
        
    # Convert permutation to Lehmer code
    lehmer = []
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            if perm[j] < perm[i]:
                count += 1
        lehmer.append(count)
        
    # Convert Lehmer code to integer payload
    payload = 0
    for i, l_val in enumerate(reversed(lehmer)):
        payload += l_val * math.factorial(i)
        
    # Convert integer to bits
    bits = []
    for _ in range(num_bits):
        bits.append(payload & 1)
        payload >>= 1
        
    return np.array(bits, dtype=np.uint8)

def apply_permutation(W: np.ndarray, b: np.ndarray, W_next: np.ndarray, P: np.ndarray):
    orig_W_shape = W.shape
    W_flat = W.reshape(orig_W_shape[0], -1)
    W_prime = (P @ W_flat).reshape(orig_W_shape)
    
    b_prime = P @ b if b is not None else None
        
    orig_next_shape = W_next.shape
    
    # Move C_in (axis 1) to the last axis
    W_next_moved = np.moveaxis(W_next, 1, -1)
    flat_next = W_next_moved.reshape(-1, W_next.shape[1])
    
    # Multiply by P.T on the right
    W_next_prime_flat = flat_next @ P.T
    
    # Reshape back and restore axis 1
    W_next_prime_moved = W_next_prime_flat.reshape(W_next_moved.shape)
    W_next_prime = np.moveaxis(W_next_prime_moved, -1, 1)
    
    return W_prime, b_prime, W_next_prime
