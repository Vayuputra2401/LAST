import numpy as np
import os

def read_skeleton_official(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True, max_body=4, njoints=25):
    """
    Official .skeleton file parser from NTU RGB+D repository.
    
    Args:
        file_path: Path to .skeleton file
        save_skelxyz: Whether to save 3D skeleton coordinates
        save_rgbxy: Whether to save RGB mapping coordinates
        save_depthxy: Whether to save Depth mapping coordinates
        max_body: Maximum bodies to track (default 4)
        njoints: Number of joints (default 25)
        
    Returns:
        bodymat: dictionary containing skeleton data
    """
    if not os.path.exists(file_path):
        return None
        
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    
    # read all lines into the pool to speed up, less io operation. 
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat['file_name'] = os.path.basename(file_path)
    bodymat['nframe'] = nframe
    nbody = int(datas[1][:-1]) # Initial body count (not used?)
    bodymat['nbodys'] = [] 
    bodymat['njoints'] = njoints
    
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
            
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        if cursor >= len(datas): break
        
        bodycount = int(datas[cursor][:-1])    
        if bodycount == 0:
            continue 
            
        # skip the empty frame 
        bodymat['nbodys'].append(bodycount)
        
        # We need to read 'bodycount' bodies, but only store up to 'max_body'
        for body in range(bodycount):
            cursor += 1
            if cursor >= len(datas): break
            
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            
            njoints_read = int(datas[cursor][:-1])
            
            # Store data if within max_body limit
            if body < max_body:
                skel_body = 'skel_body{}'.format(body)
                rgb_body = 'rgb_body{}'.format(body)
                depth_body = 'depth_body{}'.format(body)
            
            for joint in range(njoints_read):
                cursor += 1
                if body < max_body:
                    jointinfo = datas[cursor][:-1].split(' ')
                    jointinfo = np.array(list(map(float, jointinfo)))
                    
                    if save_skelxyz:
                        bodymat[skel_body][frame,joint] = jointinfo[:3]
                    if save_depthxy:
                        bodymat[depth_body][frame,joint] = jointinfo[3:5]
                    if save_rgbxy:
                        bodymat[rgb_body][frame,joint] = jointinfo[5:7]
                        
    # prune the abundant bodys 
    for each in range(max_body):
        # If this body index was never reached/filled (or only partially?), 
        # The logic in original code: "if each >= max(bodymat['nbodys'])"
        # This implies if specific body index exceeds the maximum number of bodies seen in any frame.
        if len(bodymat['nbodys']) > 0 and each >= max(bodymat['nbodys']):
            if save_skelxyz:
                if 'skel_body{}'.format(each) in bodymat:
                    del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                 if 'rgb_body{}'.format(each) in bodymat:
                    del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                 if 'depth_body{}'.format(each) in bodymat:
                    del bodymat['depth_body{}'.format(each)]
                    
    return bodymat

def convert_official_to_numpy(bodymat, max_frames=300, max_bodies=2):
    """
    Convert the official dictionary format to standard numpy format used in our repo.
    
    Args:
        bodymat: Output from read_skeleton_official
        max_frames: Target frame count
        max_bodies: Target body count
        
    Returns:
        data: (C, T, V, M)
    """
    if bodymat is None:
        return None
        
    nframe = bodymat['nframe']
    njoints = bodymat['njoints']
    
    # Target shape: (T, V, C, M)
    # Our loader uses M=2 (max_bodies) usually
    data = np.zeros((nframe, njoints, 3, max_bodies), dtype=np.float32)
    
    # Fill data
    for m in range(max_bodies):
        key = f'skel_body{m}'
        if key in bodymat:
            # (T, V, 3)
            body_data = bodymat[key]
            # Copy to (T, V, 3, m)
            if body_data.shape[0] == nframe:
                 data[:, :, :, m] = body_data
            else:
                 # Should not happen if pre-allocated correctly
                 pass
                 
    # Temporal Processing (Crop/Pad)
    # FIX: Use repeat-padding (replicate last real frame) instead of zero-padding.
    # Zero-padding bakes an artificial "zero pose" into the saved .npy files.
    # The velocity stream then has a large spike at the real-last-frame boundary:
    #   v[T-1] = J[T] - J[T-1] = 0 - J[T-1] = -J[T-1]
    # This corrupts the velocity stream for all short sequences.
    # Repeat-padding keeps v[t] ≈ 0 in the padded region (last frame repeated →
    # finite difference ≈ 0), which is semantically neutral (subject frozen).
    T = nframe
    if T < max_frames:
        last_frame = data[T - 1:T]          # (1, V, 3, M)
        repeats = max_frames - T
        pad = np.repeat(last_frame, repeats, axis=0)  # (repeats, V, 3, M)
        data = np.concatenate([data, pad], axis=0)    # (max_frames, V, 3, M)
    elif T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        data = data[indices]
        
    # (T, V, C, M) -> (C, T, V, M)
    data = data.transpose(2, 0, 1, 3)
    
    return data
