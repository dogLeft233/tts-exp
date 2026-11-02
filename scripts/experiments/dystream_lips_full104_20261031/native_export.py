"""Account for the unchanged official MoviePy exporter's floating-point clock."""
def exported_frames(rendered_frames):
    import numpy as np
    assert isinstance(rendered_frames,int) and rendered_frames>0
    # Installed MoviePy Clip.iter_frames uses this exact time grid. For249
    # frames, duration9.96 yields250 timestamps and repeats the decoded tail.
    count=len(np.arange(0,rendered_frames/25,1.0/25))
    assert rendered_frames<=count<=rendered_frames+1
    return count
