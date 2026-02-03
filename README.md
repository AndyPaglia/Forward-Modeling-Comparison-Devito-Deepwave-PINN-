# Forward-Modeling-Comparison-Devito-Deepwave-PINN-
Forward Modeling implemented using three different approaches: Devito's library, Deepwave's library and PyTorch Custom finite difference representation for Wave Propagation (to implement the Full-Waveform inversion using Physical Informed Neural Networks)

WHAT HAVE I CORRECTED?
1) COLORBAR ADDED TO EACH PLOT
2) USED THE SAME AXIS (RECEIVER INDICES)
3) USED SAME SCALE FOR EACH PLOT
4) SAME AMOUNT OF RECEIVER WHEN A SPECIFIC APERTURE IS DEFINED (IF APERTURE 4000 --> 8000/25 --> 320 RECEIVERS TOTAL, IF APERTURE 6000 --> 12000/25 --> 480 RECEIVERS)
5) PHASE INVERSION CORRECTED

TO DO:
1) FORWARD MODELING USING PINN
2) CHECK THE ADDED RINGING IN THE PINN CASE AND CHECK BLUNTNESS
