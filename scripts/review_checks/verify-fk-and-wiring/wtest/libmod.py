def build():
    from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX
    from kinetic_augment.body_model.smplx_to_mp import SMPLXToMediaPipe
    return MediaPipeToSMPLX(), SMPLXToMediaPipe()
